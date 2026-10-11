"""chat_router.py — understand a free-text question about a drug or chemical.

The Ask tab of the Drug & Chemical Explorer is a chat, but there is no AI model
behind it. This module is a deterministic, rule-based query parser plus a small
BM25 sentence ranker: no Streamlit, no network, no LLM/API.

    parse_query("how does dextromethorphan stop me from coughing")
      → entities = [dextromethorphan (Drug Library)]
        intents  = ["mechanism"]
        focus    = ["coughing"]   (ranked against the record's sentences, with
                                   lay→medical synonyms: cough → antitussive,
                                   cough reflex, medulla …)

What it recognises
* the drug/chemical — Drug Library names (RxNorm ingredients), US↔INN aliases
  (paracetamol ↔ acetaminophen), Singapore brand names (HSA register), SMILES,
  and typos (edit distance); anything else is passed on as free text for the
  app's PubChem → OPSIN → brand → CIR resolver;
* what is being asked — ~20 intents (mechanism, pharmacokinetics, dosing, side
  effects, interactions, structure, properties …) by keyword patterns;
* follow-ups — "what about its half-life?" reuses the previous drug, "and
  morphine?" reuses the previous question;
* drug classes ("what are statins?") and conditions ("what can I take for a
  cough?") answered from the Drug Library's WHO ATC classification.
"""
from __future__ import annotations

import difflib
import math
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import drug_library as dl
    _HAS_DL = dl.is_available()
except Exception:  # noqa: BLE001 — the router still works on free text
    dl = None
    _HAS_DL = False

try:
    import sg_register as sgr
    _HAS_SG = sgr.is_available()
except Exception:  # noqa: BLE001
    sgr = None
    _HAS_SG = False

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:  # noqa: BLE001
    Chem = None

try:
    import question_bank as _qb      # question-type catalogue (data + form rules)
except Exception:  # noqa: BLE001 — the router still works without it
    _qb = None

MAX_QUERY_LEN = 500

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# Function words — never focus terms, never drug names.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren as at be
because been before being below between both but by can cannot could couldn did
didn do does doesn doing don down during each either else even ever every few
for from further get gets got had hadn has hasn have haven having he her here
hers herself him himself his how i if in into is isn it its itself just let ll
me might mightn more most must mustn my myself no nor not now of off on once
one only or other our ours ourselves out over own re s same shall shan she
should shouldn so some such t than that the their theirs them themselves then
there these they this those through to too under until up upon us ve very was
wasn we were weren what whats when where whether which while who whom whose why
will with won would wouldn y yes yet you your yours yourself yourselves
d m o ain anyone anything someone something everyone everything
""".split())

# Words that carry the question's *shape* rather than its subject — dropped from
# the focus terms used to rank sentences.
FILLER = frozenset("""
drug drugs medicine medicines medication medications med meds pill pills tablet
tablets capsule capsules take takes taking took taken use uses using used work
works working worked body system effect effects thing things stuff really
actually exactly way ways much many know tell explain please give show find
need want like okay ok kind type sort about info information question answer
mean means happen happens happening safe make makes made good bad best better
long short time times lot lots pretty quite person people someone patient
patients doctor pharmacist drugx compound chemical substance molecule
difference differences between compare versus list some examples example name
names number numbers anything everything
""".split())

# Common English (and health-chat) words: never treated as a drug/brand name or
# fuzzy-matched to one. Not exhaustive — it guards the typo-correction and
# brand-name matching, which are the two places a plain word could be mistaken
# for a medicine.
COMMON_WORDS = frozenset("""
able across act action actions active add added addict addiction adult adults
advice affect affects after afternoon age aged ago agree ahead aid aids alert
allow almost alone along already alright always amount anger angry animal
another answer anxiety anxious anyway apart appear apply area arm arms around
arrive art ask asked asking asleep attack attention average avoid awake away
baby back bad bag balance ball bar base basic basis bath bear beat beautiful
became become becomes bed beer before begin beginning behind believe belly
benefit benefits best better beyond big bigger bit bite bitter black bladder
bleed bleeding blind block blocked blood blow blue board body bone bones book
born both bottle bottom bowel brain brand brands bread break breakfast breast
breath breathe breathing bright bring broken brother brown build burn burning
business busy buy call called calm came cancer candid cannot capsule car care
careful carry case cases cat catch cause caused causes causing cell cells
center central centre certain chance change changed changes cheap check chest
chew child children choice choose chronic city class clean clear clinical close
cloud coffee cold colds colour color combine come comes coming common compare
condition conditions continue control cool cope copy core correct cost costs
cough coughing coughs could count country course cover cramp crazy cream create
cure cut daily damage danger dark daughter day days dead deal dear death
decide deep depression describe design detail details develop diet difference
different difficult dinner direct dirty disease diseases dizzy doctor dog dollar
done door dose doses double down dream dress drink drinking drive drop drops dry
due during dust ear early ears easy eat eating edge effect effective eight
either elderly else empty end energy enjoy enough enter entire equal error
especially even evening event ever every exact example except exercise exist
expect expensive experience extra eye eyes face fact fail fair fall family far
fast fat father fear feel feeling feet female fever few field fight figure fill
final find fine finger finish fire first fish fit five fix flat floor flu fly
follow food foot force forget form forward four free fresh friend front fruit
full fun future gain game general gentle girl give given glad glass goes going
gold gone good great green ground group grow guess gut hair half hand happen
happy hard harm head headache health healthy hear heard heart heat heavy held
hello help helps high higher history hit hold hole home hope horse hospital hot
hour hours house however huge human hunger hungry hurt hurts idea ill illness
image important include including increase indeed inside instead interest into
iron issue item itch itchy job join joint joints keep kept key kid kids kill
kind kitchen knee knew know known lack lady land large last late later laugh
law lay lead learn least leave left leg legs less let letter level life light
like likely line lip list listen little live liver living local long look
looking loose lose loss lost lot loud love low lower lunch lung lungs main major
make male man manage many mark market matter maybe meal mean meaning measure
medical meet member memory men mental mess method middle might mild milk mind
minute miss mistake mix model moment money month months mood morning mother
mouth move much muscle muscles music name names natural nature near nearly neck
need nerve nerves never new news next nice night nine noise none normal nose
note nothing notice number nurse object odd offer office often oil old once
open opinion option options orange order organ original others outside over
overall pack page pain painful pair paper parent part party pass past patch
patient pay peace people per perhaps period person phone pick picture piece
place plan plant plastic play please plenty point poor popular possible pound
power practice prefer pregnant present press pressure pretty prevent price
private problem problems process produce product products program protect proud
public pull pump pure purpose push put quick quickly quiet quite race rain raise
range rare rash rate rather reach read ready real reason receive recent record
red reduce regular relief remain remember remove repeat reply report rest result
return rich ride right ring rise risk road rock role room root rough round rule
run sad safe safety salt same save say scale school sea season seat second see
seem seen self sell send sense serious serve service set seven several severe
shake shape share sharp shot show shower sick side sign silver simple since
single sister sit site six size skin sleep sleepy slow small smell smile smoke
smoking snack soft soil solid son soon sore sorry sort sound soup source space
speak special speed spend spirit spot spray spring square stage stand standard
star start state stay steady step stick still stomach stone stop store story
straight strange street stress strong student study stuff style subject success
sudden sugar suggest suit summer sun supply support sure surface surgery sweet
swell swelling swim symptom symptoms table tablet talk tall taste tea teach team
tear teeth temperature ten term test than thank thanks thick thin think third
thirsty though thought three throat throw tight till tired today together told
tomorrow tone tongue tonight tooth top total touch tough toward town toxic
track trade travel treat treated treating treatment tree trial trip trouble
true trust truth try trying turn twice two type ugly unable understand unit
until upper upset urine useful usual usually value various view visit voice
vomit wait wake walk wall war warm wash watch water wave way weak wear weather
week weekend weeks weight welcome well went wet whatever wheat white whole wide
wife wild win wind window wine winter wish within without woman women wonder
word words work world worry worse worst worth write wrong yard year years
yellow young youth zero zone gives given given giving makes making made
eye drop drops syrup injection injections inhaler ointment gel lotion powder
chewable liquid oral nasal topical generic version versions dosage forms form
strength strengths prescription counter shelf pharmacy hospital clinic store
online singapore america american british sg us uk usa australia malaysia
body brain heart liver kidney kidneys stomach lungs skin blood bones muscles
nerves eyes ears throat chest back head teeth gums joints bladder bowel gut
morning evening night daily weekly monthly hourly
cancer diabetes asthma arthritis migraine allergy allergies infection
infections virus viruses bacteria fungus flu covid cold fever pain cough
headache nausea vomiting diarrhea diarrhoea constipation insomnia anxiety
depression stress acne eczema rash itching hives heartburn reflux ulcer ulcers
dementia epilepsy seizure seizures stroke obesity overweight
alcohol beer wine coffee tea grapefruit juice milk food meal meals caffeine
pregnancy pregnant breastfeeding breast feeding nursing baby babies child kids
elderly older adult adults teen teenager teenagers
side effect effects adverse reaction reactions interaction interactions
mechanism action actions structure structures formula molecule molecules
property properties weight point boiling melting solubility soluble
half life lifetime metabolism metabolised metabolized excretion absorption
brand brands generic name names class classes category type types group
groups family families example examples list lists similar alternative
alternatives difference differences compare comparison versus vs
dose doses dosing dosage overdose overdoses toxic toxicity poison poisoning
warning warnings precaution precautions danger dangerous risk risks harmful
history discovered invented made produced synthesis synthesised synthesized
storage stored store expire expiry expired fridge temperature
spectrum spectra legal illegal controlled addictive addiction abuse
""".split())

# Side-effect, ingredient and everyday-question words: never a drug name and
# never typo-corrected into one ('lactose' must not become the enzyme lactase,
# 'drowsiness' must not be looked up as a chemical).
COMMON_WORDS = COMMON_WORDS | frozenset("""
drowsiness sleepiness somnolence fatigue tiredness dizziness lightheadedness sweating
sweats tremor tremors shaking palpitations bloating gas flatulence cramping bruising
hallucinations confusion agitation restlessness irritability nightmares sedation
impotence tinnitus jaundice hypoglycemia hypoglycaemia hypotension hypertension
tachycardia bradycardia edema oedema numbness tingling weakness thirst dehydration
blurred vision photosensitivity sunburn indigestion dyspepsia hiccups yawning
nosebleeds bleeds suicidal thoughts libido dependence dependency tolerance
lactose gelatin gelatine gluten halal kosher vegan vegetarian excipient excipients
filler fillers dye dyes preservative preservatives inactive ingredient ingredients
peanut soy dairy pork porcine bovine animal
differ differs anymore thinking suddenly abruptly taper tapering turkey wean
trimester trimesters sperm infertility machinery machines driving drove
subsidised subsidized subsidy expensive costly cheaper cheapest price prices
crush crushed crushing split splitting chew chewing swallow dissolve dissolving
capsule capsules sachet sachets half halves twice thrice
missed miss skip skipped forgot forget forgotten double
bedtime meals empty stomach
stronger strongest weaker weakest safer safest milder mildest harsher gentler faster
slower quicker longer shorter cheaper effective potent
toothache backache earache stomachache bellyache tummy menstrual cramps migraines
sinusitis hayfever tonsillitis bronchitis pneumonia gastritis heartburn arthritis
hay remedy remedies cure cures treatment treatments therapy therapies solution solutions
pens pen inhaler inhalers patch patches spray sprays drop drops
faeces feces stool stools bile react reacts reaction interfere interferes interfering
synthetic natural organic compared comparing hooked percentage tips advice regularly
uti utis runny blocked stuffy watery itchy die dying dead alive spinach greens vegetables
""".split())


def _is_common(w: str) -> bool:
    """A common English word, also in its plural ('headaches', 'remedies')."""
    if w in COMMON_WORDS:
        return True
    if w.endswith("ies") and w[:-3] + "y" in COMMON_WORDS:
        return True
    if w.endswith("es") and w[:-2] in COMMON_WORDS:
        return True
    return w.endswith("s") and w[:-1] in COMMON_WORDS

# Single-word Singapore brand names that are also everyday words or class
# names — never matched as brands.
_BRAND_EXCLUDE = frozenset("""
became candid water diane largo levin cinna haemo coxib propyl tropin indium
regro talis carin decan copan napa ural dolo indo klar rost poro olit domi sife
soov horf imin inon dac uft liza ella pred keno lycan prega vetic virest fleming
kirsty platless virless
""".split())

# Library ingredients that are everyday substances: in a question about another
# medicine they are the *interactant* ("codeine with alcohol"), not the subject.
_WEAK_SUBSTANCES = frozenset("""
alcohol ethanol caffeine water oxygen air nicotine sugar salt glucose sucrose
milk coffee tea grapefruit food helium gold silver copper iron zinc sulfur
""".split())

# All-capital tokens that RDKit happens to parse as SMILES but are acronyms.
_ACRONYMS = frozenset("""CNS PPI IBS COC OCP CBC NSC SCN CPS ICU NOS SOS CCB
COPD PCOS SSN CO NO SO ON IN OF IS OR IF SC IV IM PO OTC""".split())

# Class phrases that contain an element/ion word ('calcium channel blocker')
# — masked so 'calcium' isn't read as the drug being asked about.
_CLASS_MASK_RE = re.compile(
    r"\b(?:calcium|sodium|potassium|chloride|magnesium|iron|zinc|proton|hydrogen|"
    r"bile acid|folic acid|amino acid|nitric oxide)[- ](?:channels?|pumps?|sparing|"
    r"binders?|supplements?|chelat\w*|sequestrants?|blockers?|antagonists?|"
    r"inhibitors?|levels?|deficiency)\b|\bchannel blockers?\b|\bpump inhibitors?\b")

# Lay words → the technical terms sources use (for sentence ranking only).
LAY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "cough": ("antitussive", "cough reflex", "cough center", "cough centre", "medulla"),
    "cold": ("common cold", "rhinovirus", "upper respiratory"),
    "colds": ("common cold", "rhinovirus", "upper respiratory"),
    "flu": ("influenza", "common cold"),
    "coughing": ("antitussive", "cough reflex", "cough center", "medulla"),
    "pain": ("analgesic", "analgesia", "nociception", "nociceptive"),
    "ache": ("analgesic", "pain"), "hurt": ("pain", "analgesic"),
    "headache": ("headache", "migraine", "analgesic"),
    "fever": ("antipyretic", "pyrexia", "hypothalamus", "temperature"),
    "sleep": ("sedation", "sedative", "hypnotic", "somnolence", "insomnia"),
    "sleepy": ("sedation", "somnolence", "drowsiness"),
    "drowsy": ("drowsiness", "somnolence", "sedation"),
    "tired": ("fatigue", "somnolence", "asthenia"),
    "anxiety": ("anxiolytic", "anxiety"), "anxious": ("anxiolytic", "anxiety"),
    "depression": ("antidepressant", "mood", "serotonin"),
    "mood": ("depression", "serotonin", "mood"),
    "pressure": ("blood pressure", "hypertension", "antihypertensive"),
    "cholesterol": ("ldl", "lipid", "hmg coa", "cholesterol"),
    "sugar": ("glucose", "glycemic", "glycaemic", "insulin", "hyperglycemia"),
    "glucose": ("glycemic", "insulin", "hyperglycemia", "hba1c"),
    "diabetes": ("glycemic", "glucose", "insulin", "type 2"),
    "weight": ("body weight", "weight loss", "weight gain", "appetite", "obesity"),
    "appetite": ("satiety", "gastric emptying", "food intake"),
    "fat": ("adipose", "lipid", "weight"),
    "allergy": ("histamine", "antihistamine", "allergic"),
    "allergic": ("histamine", "hypersensitivity", "anaphylaxis"),
    "itch": ("pruritus", "histamine"), "itching": ("pruritus", "histamine"),
    "sneeze": ("histamine", "rhinitis"), "sneezing": ("histamine", "rhinitis"),
    "inflammation": ("anti-inflammatory", "prostaglandin", "cyclooxygenase", "cox"),
    "swelling": ("edema", "oedema", "inflammation"),
    "infection": ("antibacterial", "bacterial", "bactericidal", "antimicrobial"),
    "bacteria": ("antibacterial", "bacterial", "cell wall", "bactericidal"),
    "germs": ("antibacterial", "antimicrobial"),
    "virus": ("antiviral", "viral replication"),
    "stomach": ("gastric", "gastrointestinal", "gi"),
    "acid": ("gastric acid", "proton pump", "acid secretion"),
    "heartburn": ("gastric acid", "reflux", "gerd", "proton pump"),
    "nausea": ("antiemetic", "emesis", "vomiting"),
    "vomit": ("emesis", "antiemetic"), "vomiting": ("emesis", "antiemetic"),
    "diarrhea": ("antidiarrheal", "motility", "bowel"),
    "diarrhoea": ("antidiarrheal", "motility", "bowel"),
    "constipation": ("laxative", "bowel movement", "stool"),
    "heart": ("cardiac", "myocardial", "heart rate"),
    "clot": ("anticoagulant", "thrombosis", "coagulation", "platelet"),
    "clots": ("anticoagulant", "thrombosis", "coagulation", "platelet"),
    "bleeding": ("hemorrhage", "haemorrhage", "anticoagulant", "bleeding"),
    "breathe": ("bronchodilator", "bronchospasm", "airway", "respiratory"),
    "breathing": ("respiratory", "respiratory depression", "airway"),
    "asthma": ("bronchodilator", "bronchospasm", "airway"),
    "wheeze": ("bronchospasm", "bronchodilator"),
    "mucus": ("mucolytic", "expectorant", "secretions", "sputum"),
    "phlegm": ("mucolytic", "expectorant", "sputum"),
    "congestion": ("decongestant", "nasal congestion", "vasoconstriction"),
    "stuffy": ("decongestant", "nasal congestion"),
    "nose": ("nasal", "rhinitis"),
    "addiction": ("abuse", "dependence", "withdrawal", "tolerance"),
    "addictive": ("abuse", "dependence", "withdrawal"),
    "high": ("euphoria", "psychoactive"),
    "liver": ("hepatic", "hepatotoxicity", "liver"),
    "kidney": ("renal", "nephrotoxicity"), "kidneys": ("renal", "nephrotoxicity"),
    "pregnant": ("pregnancy", "fetal", "teratogenic"),
    "baby": ("fetal", "infant", "pregnancy"),
    "alcohol": ("alcohol", "ethanol", "alcoholic"),
    "drink": ("alcohol", "ethanol"),
    "grapefruit": ("grapefruit", "cyp3a4"),
    "food": ("food", "meal", "fed", "fasting"),
    "hair": ("alopecia", "hair loss"),
    "seizure": ("seizures", "convulsions", "epilepsy", "anticonvulsant"),
    "erection": ("erectile dysfunction", "pde5"),
    "migraine": ("migraine", "triptan", "headache"),
    "muscle": ("muscle relaxant", "myalgia", "skeletal muscle"),
    "memory": ("cognition", "cognitive", "amnesia"),
    "sex": ("libido", "sexual dysfunction"),
    "brain": ("central nervous system", "cns", "cerebral"),
    "blood": ("plasma", "serum"),
    "skin": ("dermal", "cutaneous", "topical"),
    "eye": ("ocular", "ophthalmic", "intraocular"),
    "eyes": ("ocular", "ophthalmic"),
    "dizzy": ("dizziness", "vertigo"), "dizziness": ("vertigo",),
    "rash": ("rash", "dermatitis", "stevens-johnson"),
    "stop": (), "work": (),
}

# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------
# A "how does X <verb>" question asks how the drug works.
_HOW_MECH = (r"\bhow (?:exactly |actually |precisely )?(?:does|do|did|can|could|would|will|is|are)\b"
             r"(?! (?:long|much|many|often|fast|quickly|soon|safe|bad|common|likely|"
             r"strong|effective|well|big|old)\b)[^?.!]{0,80}?\b"
             r"(?:work|works|working|act|acts|stop|stops|help|helps|relieve|relieves|"
             r"reduce|reduces|lower|lowers|raise|raises|increase|increases|decrease|"
             r"decreases|treat|treats|prevent|prevents|block|blocks|kill|kills|fight|"
             r"fights|suppress|suppresses|control|controls|ease|eases|cure|cures|calm|"
             r"calms|improve|improves|numb|numbs|thin|thins|clear|clears|dilate|"
             r"dilates|relax|relaxes|function|functions)\b")

# (intent, strong pattern, weak pattern). Weak matches only count when no strong
# intent matched (e.g. "is it safe" → warnings, but "is it safe in pregnancy"
# → pregnancy alone).
_INTENT_SPECS: List[Tuple[str, str, str]] = [
    ("mechanism",
     _HOW_MECH + r"|\b(?:mechanisms?(?: of action)?|moa|modes? of action|pharmacodynamics?|"
     r"pharmacology|pharmacological (?:action|effects?)|how (?:it|drugx|they) works?|"
     r"what does (?:it|drugx) (?:actually |really |exactly )?do\b(?! to)|what (?:it|drugx) does|"
     r"molecular targets?|targets?|"
     r"receptors?|binds?|binding|agonis[tm]s?|antagonis[tm]s?|inhibits?|inhibition|inhibitors?|"
     r"blocks?|blockers?|what (?:enzyme|receptor|target|protein|channel)s? (?:does|do) (?:it|drugx) "
     r"(?:inhibit|block|target|bind|act|affect)\w*|"
     r"(?:have|has|exerts?|produces?) its effects?|"
     r"acts? (?:on|as|at)|works? on|site of action)\b",
     r"\bhow (?:does|do|can|would|will) (?:it|drugx) (?:affect|make|change|do)\b|"
     r"\bwhat does (?:it|drugx) do to\b"),
    ("pk",
     r"\b(?:pharmacokinetics?|pk|adme|half[- ]?li(?:fe|ves)|t ?1/2|elimination|"
     r"eliminated|excret\w*|clearance|cleared|metaboli[sz]\w*|metabolites?|broken down|"
     r"break(?:s|ing)? (?:it |drugx )?down|cyp ?\d\w*|cytochrome|absorb\w*|absorption|"
     r"bioavailab\w*|onset|kicks? in|takes? effect|peak (?:levels?|concentrations?|plasma|"
     r"effect)|time to peak|(?:does|do|will) (?:it|drugx) peak|peaks?|prodrugs?|"
     r"protein[- ]bind\w*|protein[- ]bound|volume of distribution|distribut\w*|"
     r"blood[- ]brain barrier|cross(?:es)? the (?:blood|placenta)|duration of action|"
     r"how long (?!should|(?:do|can|must|may|should) (?:i|you) (?:need to |have to )?"
     r"(?:take|use|keep|stay|be on|continue)|(?:is|does) (?:a |the )?(?:course|treatment))"
     r"(?:does|do|will|is|can|until|before|for|it)[^?.!]{0,60}|"
     r"(?:body|liver|kidneys?) (?:process\w*|handle\w*|metaboli\w*|breaks? (?:\w+ )?down|"
     r"do(?:es)? (?:to|with))|process(?:es|ed)? (?:it|drugx)|get(?:s)? rid of|"
     r"removed? from (?:the|my|your) (?:body|system)|clear(?:s|ed)? (?:from|out of)|"
     r"(?:go|goes|travel\w*|move\w*) (?:in|through|around) (?:the|my|your) body|"
     r"get(?:s)? into (?:the )?(?:brain|blood|bloodstream|tissues?|cells?)|"
     r"reach(?:es)? the brain|happens? (?:to (?:it|drugx) )?in (?:the|my|your) body|"
     r"in the body after|body do with|urine tests?|false positives?|"
     r"show (?:up )?(?:on|in) (?:a )?(?:drug|urine|blood) tests?|fail (?:a )?(?:drug|urine) tests?|"
     r"feel the effects?|notice (?:the )?(?:effects?|a difference)|"
     r"before (?:it|drugx) (?:works?|starts? working|kicks? in|takes? effect)|"
     r"stays? in (?:my|your|the) (?:system|body|blood|urine)|drug tests?|"
     r"(?:start(?:s|ing)?|begins?|beginning) (?:to )?work\w*|takes? to work|wears? off|"
     r"leaves? (?:the|my|your) (?:body|system)|out of (?:my|your|the) (?:system|body)|"
     r"through (?:the )?urine|in (?:the )?urine|"
     r"how (?:fast|quickly|soon) (?:does|do|will|can|is|until))\b",
     ""),
    ("uses",
     r"\b(?:used (?:for|to|in|as)|uses?|what (?:is|are) (?:it|drugx|they) (?:for|used for|"
     r"prescribed for|good for)|indications?|indicated|prescribed (?:for|to)|good for|"
     r"helps? (?:with|for)|purpose|why (?:do|would|is|are|did)[^?.!]{0,40}"
     r"(?:take|prescribe\w*|use|given)|(?:can|does|will|is|could) (?:it|drugx)[^?.!]{0,20}"
     r"\b(?:treat|cure|help)|to treat|treats|treatment (?:of|for)|therapeutic|"
     r"approved (?:for|to)|what conditions?|what'?s (?:it|drugx) (?:for|used for|good for)|"
     r"when (?:is|are) (?:it|drugx|they) (?:used|prescribed|given)|works? for|"
     r"effective (?:for|against|in treating)|"
     r"(?:take|use) (?:it|drugx) for (?!\d|a (?:week|month|day)|long|how long|more than|over|years|"
     r"months|weeks|days))\b",
     ""),
    ("dosing",
     r"\b(?:dos(?:e|es|age|ages|ing)|"
     # 'how much/many' — but not 'how much of it is protein bound', 'how many
     # people…', 'how many stereocentres', 'how much does it cost'
     r"how (?:much|many)(?! (?:of )?(?:drugx|it) (?:is|are|gets?) (?:protein|bound|absorbed|"
     r"excreted|metaboli\w*|eliminated|cleared)| people| patients| users| stereo\w*| chiral\w*"
     r"| atoms?| rings?| carbons?| bonds?| hydrogens?| times| hours| days| weeks| years|"
     r" (?:does|do|is|are|would|will) (?:it|drugx|they)(?: (?:cost|be|in|at))?\b|"
     r" (?:is|are|does|do) (?:a |one |the )?(?:box|pack|packet|bottle|strip|course|tube)s?\b|"
     r" (?:of )?(?:drugx|it) (?:is|would be|can be|could be) (?:dangerous|toxic|lethal|fatal|deadly))|"
     r"max(?:imum)? (?:daily )?(?:dose|amount|intake)|"
     r"how often|how frequently|how (?:do|should|can|must|"
     r"would) (?:i|you|we|one) (?:take|use|give|apply|administer)|how long should|"
     r"how long (?:do|can|must|may) (?:i|you) (?:take|use|keep)|maximum (?:daily )?"
     r"(?:dose|amount)|max(?:imum)? (?:per|a) day|per day|times (?:a|per) day|"
     r"max(?:imum)?\b[^?.!]{0,30}\b(?:a|per|in a|each|every) (?:day|24 hours)|in 24 hours|"
     r"most (?:drugx )?(?:i|you|one) (?:can|could|should) take|max (?:i|you) can take|"
     r"(?:once|twice|three times|four times) (?:a|per) day|every \d+(?: ?(?:to|-) ?\d+)? hours|"
     r"hours apart|"
     r"directions|administ\w*|regimen|strengths?|dosage forms?|formulations?|"
     r"what forms?|forms? (?:is|does|of)|come (?:as|in) (?:a |an )?\w+|syrups?|liquid form|"
     r"mg|mcg|milligrams?|miss(?:ed|ing)? (?:a |my |one |the )?(?:dose|pill|tablet)s?|"
     r"skip(?:ped|ping)? (?:a |my )?(?:dose|pill|tablet)|forg[eo]t(?:ten)? (?:to take )?|"
     r"double up|when (?:to|should i) take|"
     r"crush\w*|split\w*|cut (?:it |them |drugx )?in half|chew\w*|swallow(?:ed)? (?:it |them )?whole|"
     r"open (?:the |drugx )?capsules?|break (?:the |it )?(?:tablets?|pills?)|"
     r"how (?:should|must|is|are|can|do) (?:i |you |it |drugx )?(?:be )?taken|how to take|"
     r"empty stomach|with or without (?:food|meals?)|before or after (?:food|meals?|eating)|"
     r"(?:before|after) (?:meals?|food|eating|breakfast|bed)|food affect\w*|"
     r"morning or (?:at )?night|(?:in the )?morning|at night|bedtime|time of day|best time|"
     r"what time (?:should|do|can|must|to|would)\b|long[- ]term|every ?day|daily basis|"
     r"for \d+ (?:days?|weeks?|months?)|for (?:a|one|two) (?:week|month)s?|"
     r"(?:course|treatment) (?:length|duration)|how many (?:days|weeks) (?:should|do|must|can|will) (?:i|you)|"
     r"(?:usual|normal|typical|standard|recommended|right|correct|safe|starting) (?:amount|quantity|dose|"
     r"dosage)|daily limit|upper limit|limit (?:for|of|per)|highest dose|how regularly|hours between doses|"
     r"in the (?:evening|afternoon)|eat (?:before|after|with)|without food|with food|after food|before food|"
     r"tips (?:for|on) (?:taking|using)|advice (?:on|for) (?:taking|using)|"
     r"(?:best|right|correct|proper) way to take|"
     r"(?:can|should|may|do) (?:i|you) dissolve (?:it |drugx |them )?in (?:water|juice|milk)|"
     r"stay on (?:it|drugx)|be on (?:it|drugx)|for (?:years|months|life|a long time|long periods?)|"
     r"course of|available (?:as|in) (?:an? )?(?:tablets?|capsules?|liquid|syrup|injections?|creams?|"
     r"patch(?:es)?|inhalers?|sprays?|suppositor\w*|drops|gels?|suspensions?|solutions?)|"
     r"(?:liquid|chewable|dissolvable|injectable|syrup|topical|cream) (?:version|form|formulation)s?)\b",
     ""),
    ("side_effects",
     r"\b(?:side[- ]?effects?|adverse (?:effects?|reactions?|events?)|adrs?|"
     r"unwanted effects?|reactions?|(?:does|can|will|could|may|might|do) (?:it|drugx|they)"
     r"[^?.!]{0,30}?\bcause|causes?|caused|causing|make (?:me|you|people|someone|us) "
     r"(?:feel )?\w+|makes? (?:me|you) feel|feel (?:sick|sleepy|drowsy|tired|dizzy|weird|"
     r"funny|bad)|common effects?|bad effects?|long[- ]term effects?|"
     r"(?:will|would|can|could|does|did|might|may) (?:it|drugx|they) give (?:me|you|people|"
     r"someone|us)|lead(?:s|ing)? to|result(?:s|ing)? in|brings? on|"
     r"how (?:common|likely|rare|frequent(?:ly)?) (?:is|are|do|does)|"
     r"how many (?:people|patients|users) (?:get|experience|have|develop|report)|"
     r"chances? of (?:getting|having)|incidence of|(?:sex|sexual) drive|libido|"
     # 'does it raise blood sugar' / 'will I gain weight on it' — an effect
     # question (the verdict also checks whether it is the INTENDED effect)
     r"(?:does|do|can|will|could|would|might|may) (?:it|drugx|they) (?:raise|increase|lower|"
     r"decrease|reduce|elevate|worsen|affect|change|spike|drop|thin|damage|harm)|"
     r"(?:gain|gaining|put on|putting on|lose|losing) weight|weight (?:gain|loss)|"
     r"(?:from|caused by|of|on) (?:drugx|it) (?:go away|wear off|last|stop|subside|settle|ease)|"
     r"(?:what will|how will) (?:it|drugx) (?:do to|make) (?:me|my body|you)|"
     r"what (?:will )?happens? to (?:me|my body|you) (?:on|if i take|when taking|after taking|while taking)|"
     r"(?:common|rare|normal|frequent|usual|expected) (?:with|on|from|when taking)|linked to|"
     r"associated with|percentage of (?:people|patients)|hard on (?:the |my |your )?\w+|"
     r"(?:hurt|harm|damage|upset|irritate)s? (?:my|your|the) \w+|what to expect|"
     r"non[- ]?drowsy|non[- ]?sedating|drowsy|sedating|"
     r"first (?:few )?(?:days|weeks?) (?:on|of|taking)|first (?:few )?(?:days|week))\b",
     ""),
    ("warnings",
     r"\b(?:warnings?|precautions?|boxed warning|black[- ]box|contraindicat\w*|"
     r"who (?:should|shouldn'?t|should not|can'?t|cannot|can not|must not|mustn'?t) "
     r"(?:take|use|have)|when not to|(?:should|must) (?:it|drugx) not be|allerg\w*|"
     r"avoid (?:if|in)|drive|driving|operate (?:heavy )?machinery|machinery|"
     r"stop(?:ping|ped)? (?:taking |using )?(?:it|drugx|my drugx)|"
     r"(?:quit|quitting) (?:taking )?(?:it|drugx)|(?:come|coming|get|getting) off|wean\w* off|"
     r"cold turkey|taper\w*|discontinu\w*|suddenly stop\w*|"
     r"(?:kidney|renal|liver|hepatic) (?:disease|problems?|failure|impairment|damage|issues?)|"
     r"not suitable|unsuitable|(?:should|must) avoid (?:it|drugx)|conditions? (?:that )?(?:should )?avoid|"
     r"know before (?:taking|starting|using)|before (?:taking|starting|using) (?:it|drugx)|"
     r"bad for (?:my |your |the )?(?:kidneys?|liver|heart|stomach|bones|brain|health)|"
     r"careful (?:of|about|with)|watch out(?: for)?|look out for|be aware of|"
     r"(?:ok|okay|safe|fine|alright) (?:with|for people with|if (?:i|you) have) (?:my |a |an )?"
     r"(?:high blood pressure|hypertension|asthma|diabetes|heart (?:disease|problems?|failure)|kidney|"
     r"liver|ulcers?|epilepsy|glaucoma|gout|low blood pressure)|"
     r"before (?:surgery|an operation|a procedure|dental (?:work|treatment)|an? (?:x-?ray|ct scan|mri)|"
     r"contrast))\b",
     r"\b(?:safe|safely|safety|dangerous|danger|risks?|risky|harmful|bad for|"
     r"is it ok|okay to|can i (?:take|use|have)|should i (?:take|use))\b"),
    ("interactions",
     r"\b(?:interact\w*|together|combin\w*|mix(?:ed|ing)?|at the same time|alongside|"
     r"along with|concomitant\w*|co-?administ\w*|with (?:alcohol|food|grapefruit|milk|"
     r"coffee|caffeine|other (?:drugs|medicines|medications|meds))|"
     r"(?:drink|drinking) (?:alcohol|beer|wine|coffee)|grapefruit|alcohol|booze|"
     r"wine|beer|liquor|whisk(?:e)?y|vodka|rum|gin|soju|cocktails?|"
     r"avoid (?:while|when|with|on)\b|(?:what|things) (?:to|should i|i should) avoid|"
     r"(?:foods?|drinks?|things|medicines?|medications?|drugs?) (?:to|should i|i should|i must) avoid|"
     r"not (?:to )?(?:take|be taken|be mixed|mix|combine|be combined|be used|use) with|"
     r"birth control|the pill|(?:oral )?contraceptives?|contraception|"
     r"drink (?:on|while|when|with|after|alcohol)|have a drink|drinking (?:on|while|when)|"
     r"(?:can|should|could) (?:i|you|she|he|they|we) (?:still )?eat\b|eat(?:ing)? (?:\w+ ){0,2}(?:while|on) (?:it|drugx)|"
     r"react(?:s|ion)? with|interfere(?:s)? with|shouldn'?t (?:i )?(?:take|mix|combine) (?:\w+ )?with|"
     r"interact with other (?:medicines|medications|drugs))\b",
     ""),
    ("pregnancy",
     r"\b(?:pregnan\w*|breast[- ]?feed\w*|breastfeed\w*|breast[- ]?milk|lactat\w*|"
     r"nursing (?:mothers?|moms?)|while nursing|when nursing|trying to conceive|conceiv\w*|"
     r"fertility|infertil\w*|sperm|trimesters?|"
     r"unborn|fetus|foetus|fetal|foetal|teratogen\w*|birth defects?)\b",
     ""),
    ("children",
     r"\b(?:child|children|kids?|pediatric|paediatric|infants?|toddlers?|bab(?:y|ies)|"
     r"newborns?|teen(?:s|agers?)?|adolescents?|under (?:\d+|twelve|six|two)|"
     r"\d+ ?(?:year|yr|month|mo|week|wk)s?[- ]?old)\b",
     ""),
    ("elderly",
     r"\b(?:elderly|older (?:adults?|people|patients|persons)|geriatric|seniors?|"
     r"old age|over 65|aged (?:6|7|8|9)\d)\b",
     ""),
    ("overdose",
     r"\b(?:overdos\w*|too much|too many|toxic\w*|poison\w*|ld ?50|lethal|fatal|deadly|"
     r"kill (?:you|me|someone|a person)|antidote|swallowed|ingested|double dose|"
     r"took (?:\d{2,}|a whole|the whole|all (?:of )?(?:the|my|his|her))|"
     r"took (?:an? )?(?:extra|additional|double)|accidentally (?:took|swallowed|ingested|gave|had)|"
     r"how much (?:of )?(?:drugx|it) (?:is|would be|can be|could be) (?:dangerous|toxic|lethal|fatal|deadly)|"
     r"(?:ate|eaten|got into) (?:some |a |the |my |his |her )?(?:drugx|pills?|tablets?|medicine))\b",
     ""),
    ("brands",
     r"\b(?:brand(?:s|ed| names?)?|trade ?names?|sold (?:as|under)|marketed (?:as|under)|"
     r"generic (?:names?|versions?|equivalents?)|other names?|also (?:known|called) as|"
     r"aka|over[- ]the[- ]counter|otc|prescription[- ]only|need (?:a )?prescription|"
     r"without (?:a )?prescription|"
     r"(?:can|where (?:can|do)) (?:i|you) (?:buy|get)(?! (?:addicted|hooked|high|dependent|"
     r"sick|pregnant|used|withdrawal))|buy|"
     r"purchase|availab\w*|in singapore|singapore|hsa|registered|pharmacy[- ]only|"
     r"in sg|in the us|names? (?:of|for) (?:it|drugx)|active ingredients?|what drug is|"
     r"see a (?:doctor|gp) (?:to get|for)|need (?:a |to see a )?(?:doctor|gp)|"
     r"pharmacist (?:give|sell|recommend)|get (?:it|drugx) (?:from|at) (?:a |the )?(?:pharmacy|chemist|"
     r"pharmacist))\b",
     ""),
    ("class",
     r"\b(?:drug class|class(?:es)?|classification|classified|what (?:kind|type|sort) of "
     r"(?:drug|medicine|medication|compound|chemical|substance)|type of (?:drug|medicine|"
     r"medication)|categor(?:y|ies)|family of|drug family|group of (?:drugs|medicines)|"
     r"counts? as an? |considered an? |belong(?:s|ing)? to|(?:what|which) (?:drug )?group|"
     r"drugx (?:is|are) (?:an? |one of the )|"
     r"is (?:it|drugx) (?:an? )?(?:\w+ ){0,2}(?:opioid|nsaid|steroid|antibiotic|statin|"
     r"ssri|snri|stimulant|depressant|benzodiazepine|benzo|antihistamine|antidepressant|"
     r"narcotic|painkiller|analgesic|antipsychotic|beta[- ]blocker|ace inhibitor|"
     r"diuretic|anticoagulant|blood thinner|sedative|hormone|vitamin|vaccine|"
     r"antiviral|antifungal|laxative|decongestant))\b",
     ""),
    ("alternatives",
     r"\b(?:alternatives?|similar (?:drugs?|medicines?|medications?|to)|"
     r"(?:drugs?|medicines?|medications?) (?:like|similar)|instead of|substitutes?|"
     r"replacements?|other (?:drugs|options|medications|medicines|choices)|same class|"
     r"related drugs|(?:besides|other than|apart from) (?:it|drugx)|"
     r"what else (?:can|could|should) (?:i|you|we) (?:take|use|try))\b",
     ""),
    ("identifiers",
     r"\b(?:cas(?: number| no\.?| rn| registry(?: number)?)?|inchi(?:key)?|cid|"
     r"pubchem(?: id| cid| number)?|chembl(?: id)?|identifiers?|ids?|unii|"
     r"drugbank(?: id)?|smiles|iupac(?: name)?|systematic name|chemical name|"
     r"registry number)\b",
     ""),
    ("structure",
     r"\b(?:structures?|structural|skeletal|draw(?:ing|n)?|depict\w*|what (?:does|do) "
     r"(?:it|drugx|they) look like|looks? like|diagram|pictures?|image|molecule|"
     r"molecular (?:structure|formula|shape|model)|chemical (?:formula|structure)|"
     r"formula|3d|3-d|three[- ]dimensional|conformers?|functional groups?|chiral\w*|"
     r"stereo\w*|enantiomers?|isomers?|atoms?|bonds?|rings?|backbone|scaffold|"
     r"moiet(?:y|ies)|racemic|racemates?)\b",
     ""),
    ("properties",
     r"\b(?:molecular (?:weight|mass)|mol(?:ecular)?\.? ?wt|mw|molar mass|logp|log ?p|"
     r"lipophilic\w*|hydrophobic\w*|hydrophilic\w*|pka|pkb|acidic|(?:acid|base) or "
     r"(?:acid|base)|a (?:weak |strong )?(?:acid|base)|ioni[sz]\w*|solub\w*|dissolv\w*|"
     r"melting point|boiling point|density|vapou?r pressure|physical propert\w*|"
     r"chemical propert\w*|propert(?:y|ies)|tpsa|polar surface area|drug[- ]?likeness|"
     r"drug[- ]like|lipinski|rule of (?:5|five)|h[- ]?bond (?:donors?|acceptors?)|"
     r"hydrogen bond\w*|appearance|colou?r|odou?r|is it a (?:solid|liquid|gas)|"
     r"stability|reactiv\w*|flash point|refractive index|charge|charged|how heavy|weighs?|mass of|"
     r"basic|alkaline|body ph|physiological ph|fat[- ]soluble|water[- ]soluble)\b",
     ""),
    ("spectra",
     r"\b(?:spectr\w*|nmr|1h nmr|13c|infrared|ir spectrum|mass spec\w*|ms/ms|"
     r"uv[- ]?vis\w*|raman|chromatogra\w*|hplc|gc[- ]?ms|lc[- ]?ms)\b",
     ""),
    ("history",
     r"\b(?:history|historical|discover\w*|invent\w*|who (?:made|created|developed|found|"
     r"synthesi[sz]ed)|when was (?:it|drugx) (?:made|invented|discovered|approved|"
     r"developed|first)|origins?|developed by|first (?:approved|made|synthesi[sz]ed|used|"
     r"marketed)|approval date|year (?:of )?approv\w*|(?:when|what year) (?:\w+ ){0,2}(?:come|came) out|"
     r"introduced|first (?:sold|available))\b",
     ""),
    ("synthesis",
     r"\b(?:synthes[ie]s|synthesi[sz]\w*|how (?:is|are) (?:it|drugx|they) (?:made|produced|"
     r"manufactured|synthesi[sz]ed)|manufactur\w*|production|prepar(?:ed|ation)|"
     r"biosynthes\w*|made from|derived from|natural source|extracted from|"
     r"how (?:do|did) (?:they|you|people|companies|manufacturers) make|goes into making|"
     r"comes? from|natural or synthetic|synthetic or natural|naturally occurring|plant[- ]derived)\b",
     ""),
    ("legal",
     r"\b(?:legal(?:ity| status)?|illegal|controlled(?: substance| drug)?|"
     r"schedule (?:i{1,3}|iv|v|[1-5])|scheduled|banned|addict\w*|abus\w*|dependen\w*|"
     r"withdrawals?|habit[- ]forming|recreational(?:ly)?|get(?:ting)? (?:you |me |people )?high|"
     r"euphori\w*|street (?:names?|use)|hooked|"
     r"cause(?:s)? (?:physical |psychological )?(?:dependence|addiction|withdrawal))\b",
     ""),
    ("storage",
     r"\b(?:stor(?:e|ed|age|ing)|shelf[- ]life|expir\w*|refrigerat\w*|fridge|freezer|"
     r"room temperature|out of reach|how (?:should|do|can|to|must) (?:i |you |we )?keep|"
     r"(?:be )?kept (?:in|out|at|cool|cold)|(?:way|where) (?:to|should i|do i|can i) keep|"
     r"keep (?:it|drugx|them) (?:in|at|out|cool|cold|dry|away)|once opened|after opening|"
     r"out of date|use[- ]by|best before|past (?:its|the) (?:date|expiry))\b",
     ""),
    ("cost",
     r"\b(?:prices?|costs?|costly|expensive|cheap(?:er|est)?|afford\w*|subsid\w*|medisave|"
     r"insurance|how much (?:does|do|is|are) (?:it|drugx|they)(?: cost)?(?! (?:protein|bound|"
     r"absorbed|excreted))\b|how much (?:is|are|does|do) (?:a |one |the )?(?:box|pack|packet|bottle|"
     r"strip|course|tube)s?\b)",
     ""),
    ("ingredients",
     r"\b(?:(?<!active )ingredients?|inactive ingredients?|excipients?|fillers?|lactose|"
     r"gelatine?|gluten|halal|kosher|vegan|vegetarian|animal[- ]derived|pork|porcine|bovine|"
     r"sugar[- ]free|dyes?|colou?rings?|preservatives?|alcohol[- ]free|"
     r"contains? (?:any )?(?:sugar|alcohol|dairy|soy|peanuts?|nuts?)|"
     r"what'?s in (?:it|drugx|the|my)|what is in (?:it|drugx)|animal products?|vegetarians?|"
     r"dairy[- ]free|gluten[- ]free)\b",
     ""),
    ("overview",
     "",
     r"\b(?:what (?:is|are|was)|what's|whats|tell me (?:about|more)|overview|summary|"
     r"summari[sz]e|explain|info(?:rmation)?(?: on| about)?|describe|about|basics|"
     r"who makes)\b"),
]
_INTENT_RES = [(name, re.compile(strong) if strong else None, re.compile(weak) if weak else None)
               for name, strong, weak in _INTENT_SPECS]

INTENT_TITLES = {
    "overview": "Overview", "mechanism": "How it works", "pk": "Pharmacokinetics",
    "uses": "Uses", "dosing": "Dosing", "side_effects": "Side effects",
    "warnings": "Warnings & who shouldn't take it", "interactions": "Interactions",
    "pregnancy": "Pregnancy & breastfeeding", "population": "Children & older adults",
    "overdose": "Overdose & toxicity", "brands": "Brands & availability",
    "class": "Drug class", "alternatives": "Similar drugs", "identifiers": "Identifiers",
    "structure": "Structure", "properties": "Properties", "spectra": "Spectra",
    "history": "History", "synthesis": "How it's made", "legal": "Abuse & legal status",
    "storage": "Storage", "cost": "Cost", "ingredients": "Ingredients",
    "search": "From the record",
}

# Which ADME fact a pharmacokinetics question is about.
_PK_FACT_RES = [
    ("Onset", re.compile(r"onset|kicks? in|take\w* (?:effect|to work)|start\w* (?:to )?work|"
                         r"begin\w* (?:to )?work|how (?:fast|quickly|soon)|feel (?:it|better|the effect)|"
                         r"until i feel")),
    ("Half-life", re.compile(r"half[- ]?li|how long|\blasts?\b|\bstays?\b|duration|"
                             r"t ?1/2|drug tests?|in (?:my|your) system")),
    ("Metabolism", re.compile(r"metaboli|broken down|break\w* (?:it |drugx )?down|cyp|"
                              r"cytochrome|liver|metabolite|enzyme")),
    ("Excretion", re.compile(r"excret|eliminat|clearance|cleared|kidney|urine|renal|"
                             r"f(?:a)?eces|bile")),
    ("Absorption", re.compile(r"absor|bioavail|onset|kicks? in|take\w* effect|how (?:fast|"
                              r"quickly|soon)|peak|empty stomach|with food")),
    ("Protein binding", re.compile(r"protein")),
    ("Distribution", re.compile(r"distribut|blood[- ]brain|placenta|cross")),
]

_COMPARE_RE = re.compile(r"\b(?:vs\.?|versus|compare\w*|comparison|differen\w*|differ|"
                         r"better|worse|stronger|weaker|which (?:is|one)|or)\b")
_CONNECT_RE = re.compile(r"\b(?:with|and|plus|together|while (?:on|taking)|after)\b|[+&]")
_PRONOUN_RE = re.compile(r"\b(?:it|its|it's|this|that|they|them|their|the drug|the medicine|"
                         r"the medication|this drug|this medicine|same)\b")
_FOLLOW_ON_RE = re.compile(r"^(?:and|what about|how about|same (?:for|with)|and for|"
                           r"what of|now|then)\b")
_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|yo|hiya|howdy|good (?:morning|afternoon|evening)|help|"
    r"what can you do|what do you do|how does this work|what is this|how do (?:i|you) use (?:this|you|it)|"
    r"who are you|what are you|start|\?)(?: there| everyone| all| bot| again| me)?[\s!.?]*$")
_THANKS_RE = re.compile(r"^(?:(?:thanks?|thank you|thx|ty|cheers|ok(?:ay)?|great|cool|nice|"
                        r"awesome|perfect|got it|alright|noted|sounds good)(?: (?:you|so much|a lot))?"
                        r"[\s,!.]*){1,3}$")

# ---- safety routing (checked before anything else) ----
# First-person self-harm / suicidal intent → crisis support only, never doses.
_CRISIS_RE = re.compile(
    r"\b(?:kill(?:ing)? my ?self|end(?:ing)? (?:my|it) (?:life|all)|take my (?:own )?life|"
    r"suicid\w*|want(?:s|ed)? to die|wanna die|(?:don'?t|do not) want to (?:live|be alive|"
    r"wake up|exist)|self[- ]?harm\w*|(?:hurt|harm)(?:ing)? my ?self|overdose on purpose|"
    r"(?:want|going|plan(?:ning)?|trying) to (?:overdose|od)\b|(?:to|would|will|could) kill me\b|"
    r"(?:i|me|myself) (?:\w+ ){0,6}to die\b|how (?:much|many) [^?.!]{0,50}\b(?:to|and) die\b|"
    r"\b(?:overdose|od) (?:and|to) die\b|"
    r"(?:fatal|lethal|deadly) (?:for|to) me\b|enough to (?:kill|end) (?:me|myself|it all)\b|"
    r"no (?:reason|point) (?:to|in) (?:live|living)|better off dead)")
# Making / extracting illicit or controlled drugs → declined.
_ILLICIT_RE = re.compile(
    r"\b(?:meth|methamphetamine|crystal meth|mdma|ecstasy|molly|lsd|heroin|cocaine|fentanyl|"
    r"ghb|dmt|amphetamines?|ketamine|pcp|codeine|morphine|oxycodone|opium|psilocybin|"
    r"mescaline|thc|cannabis oil|krokodil|desomorphine|mephedrone|ephedrine|pseudoephedrine)\b")
_HOW_TO_MAKE_RE = re.compile(
    r"\b(?:how (?:to|do i|can i|do you|would i|could i|should i|do people)|ways? to|recipe|"
    r"steps? (?:to|for)|guide (?:to|for)|instructions? (?:to|for)|tutorial)\b[^?.!]{0,40}"
    r"\b(?:make|making|cook|cooking|synthesi[sz]e|synthesi[sz]ing|brew|manufacture|produce|"
    r"extract|extracting|isolate|grow|growing|refine|purify|separate|get (?:\w+ ){0,2}out of|"
    r"pull (?:\w+ ){0,2}out of)\b|\brecipe for\b|"
    r"\b(?:home ?made|homemade|diy)\b")
# Questions about giving/feeding human medicines to animals → out of scope.
_PET_RE = re.compile(r"\b(?:dogs?|pupp(?:y|ies)|cats?|kittens?|pets?|rabbits?|hamsters?|"
                     r"guinea pigs?|horses?|parrots?|birds?|ferrets?)\b")
_PET_CUE_RE = re.compile(r"\b(?:give|giving|gave|ate|eaten|swallowed|licked|chewed|got into|"
                         r"safe for|for (?:my|our|a|the)|dose for|toxic (?:to|for)|my|our)\b|"
                         r"\b(?:can|should|could|do|does) (?:my |our |a |the )?(?:dogs?|cats?|pets?|"
                         r"pupp(?:y|ies)|kittens?|rabbits?) (?:have|take|eat|get|be given)\b")
_PET_URGENT_RE = re.compile(r"\b(?:ate|eaten|swallowed|licked|chewed|got into|took)\b")
_LIST_WORDS_RE = re.compile(r"\b(?:what are|list|examples?|which (?:drugs|medicines)|types? of|"
                            r"name (?:some|a few)|all|kinds? of|drugs? (?:in|that are)|"
                            r"members?)\b")
_CONDITION_TRIGGER_RE = re.compile(
    r"\b(?:for|treat\w*|take|medicine|medicines|medication|medications|meds|drugs?|"
    r"tablets?|pills?|remed\w*|help\w*|relief|relieve|cure|options?|good)\b")

_INTERACTANT_RE = re.compile(
    r"\b(?:alcohol|beer|wine|liquor|grapefruit(?: juice)?|food|meals?|milk|dairy|coffee|"
    r"caffeine|antacids?|smoking|tobacco|cannabis|marijuana|supplements?|herbal\w*|"
    r"st\.? john'?s wort|contraceptives?|birth control)\b")

_TOKEN_RE = re.compile(r"[a-z0-9µα-ω]+")


# ---------------------------------------------------------------------------
# Lexicon (Drug Library + aliases + Singapore brands)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9µα-ω]+", " ", (s or "").lower())).strip()


# Common names → Drug Library ingredient (RxNorm) names.
_EXTRA_ALIASES = {
    "vitamin c": "ascorbic acid", "vit c": "ascorbic acid",
    "vitamin d": "cholecalciferol", "vitamin d3": "cholecalciferol",
    "vitamin b12": "cyanocobalamin", "vitamin b1": "thiamine",
    "vitamin b6": "pyridoxine", "vitamin b3": "niacin", "vitamin k": "phytonadione",
    "vitamin k1": "phytonadione", "vitamin a": "vitamin a", "vitamin e": "vitamin e",
    "thc": "dronabinol", "cbd": "cannabidiol", "asa": "aspirin",
    "acetylsalicylic acid": "aspirin", "paracetamol": "acetaminophen",
    "adrenaline": "epinephrine", "noradrenaline": "norepinephrine",
    "salbutamol": "albuterol", "lignocaine": "lidocaine", "frusemide": "furosemide",
    "pethidine": "meperidine", "glibenclamide": "glyburide",
    "rifampicin": "rifampin", "ciclosporin": "cyclosporine",
    "hyoscine": "scopolamine", "mesalazine": "mesalamine",
    "chlorphenamine": "chlorpheniramine", "guaiphenesin": "guaifenesin",
    "amoxycillin": "amoxicillin", "cefalexin": "cephalexin",
    "phenobarbitone": "phenobarbital", "oestradiol": "estradiol",
    "glyceryl trinitrate": "nitroglycerin", "gtn": "nitroglycerin",
    "dextromethorphan hbr": "dextromethorphan", "penicillin": "penicillin g",
    "insulin": "insulin", "mdma": "mdma", "ecstasy": "mdma", "lsd": "lsd",
    "dmt": "dmt", "ketamine": "ketamine", "cocaine": "cocaine",
}

# Salt / hydrate words stripped from a brand's ingredient before matching it to
# the library ('Atorvastatin Calcium' → atorvastatin).
_SALT_WORDS_RE = re.compile(
    r"\b(?:hcl|hydrochloride|dihydrochloride|hbr|hydrobromide|sodium|potassium|calcium|"
    r"magnesium|sulphate|sulfate|maleate|citrate|mesylate|besylate|tartrate|succinate|"
    r"fumarate|phosphate|acetate|bromide|hyclate|monohydrate|dihydrate|trihydrate|"
    r"hemihydrate|anhydrous|disodium|dipropionate|nitrate)\b")


@lru_cache(maxsize=1)
def _lexicon() -> Tuple[Dict[str, Dict], Dict[str, Dict], List[str]]:
    """(phrase → entry, nospace-phrase → entry, fuzzy candidate keys)."""
    lex: Dict[str, Dict] = {}
    if _HAS_DL:
        for d in dl.DRUGS:
            key = _norm(d["query"])
            if key and key not in lex:
                lex[key] = {"kind": "library", "query": d["query"], "display": d["name"],
                            "rxcui": d.get("rxcui")}
    aliases = dict(_EXTRA_ALIASES)
    if _HAS_SG:
        for us, sgs in getattr(sgr, "_ALIASES", {}).items():
            for sg in sgs:
                aliases.setdefault(sg, us)
    for alias, target in aliases.items():
        key, tkey = _norm(alias), _norm(target)
        if key in lex:
            continue
        base = lex.get(tkey)
        if base:
            lex[key] = dict(base, kind="alias", display=alias[:1].upper() + alias[1:],
                            alias_of=base["display"])
        else:
            lex[key] = {"kind": "free", "query": target,
                        "display": alias[:1].upper() + alias[1:]}
    if _HAS_SG:
        seen = set()
        for p in sgr.PRODUCTS:
            b = sgr.brand_of(p.get("name", ""))
            key = _norm(b)
            if not key or key in seen or key in lex:
                continue
            seen.add(key)
            words = key.split()
            if len(words) == 1 and (len(key) < 5 or key in COMMON_WORDS or key in _BRAND_EXCLUDE
                                    or key in STOPWORDS or not key.isalpha()):
                continue
            # a multi-word brand needs one distinctive word ('Panadol Extra'),
            # not 'Alu Tab' / 'B S' / 'Cough En'
            if not any(len(w) >= 4 and w.isalpha() and w not in COMMON_WORDS
                       and w not in STOPWORDS and w not in _BRAND_EXCLUDE for w in words):
                continue
            lex[key] = {"kind": "brand", "query": "", "display": b, "brand": b}
    nospace = {}
    for key, entry in lex.items():
        ns = key.replace(" ", "")
        if " " in key and ns not in lex:
            nospace.setdefault(ns, entry)
    # Typo correction only against ingredient names: brand names are short and
    # look alike ('Claritin' is one letter from Singapore's 'Clariwin', which
    # is clarithromycin) — an unknown brand goes to the resolver as typed.
    fuzzy = sorted(k for k, v in lex.items() if v["kind"] in ("library", "alias")
                   and " " not in k and len(k) >= 5 and k.isalpha() and k not in COMMON_WORDS)
    return lex, nospace, fuzzy


def _osa_distance(a: str, b: str) -> int:
    """Optimal-string-alignment edit distance (a transposition counts as one)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _allowed_typos(n: int) -> int:
    return 1 if n <= 7 else 2 if n <= 12 else 3


def fuzzy_candidates(token: str, limit: int = 3) -> List[Tuple[str, int]]:
    """Library/brand names within typo distance of `token` → [(key, distance)]."""
    token = token.lower()
    lex, _ns, keys = _lexicon()
    if not keys or len(token) < 5:
        return []
    close = difflib.get_close_matches(token, keys, n=8, cutoff=0.72)
    scored = sorted(((k, _osa_distance(token, k)) for k in close), key=lambda kd: (kd[1], kd[0]))
    return [(k, dist) for k, dist in scored if dist <= _allowed_typos(len(token))][:limit]


def lexicon_entry(key: str) -> Optional[Dict]:
    lex, ns, _ = _lexicon()
    return lex.get(key) or ns.get(key.replace(" ", ""))


def _brand_entity(entry: Dict) -> Optional[Dict]:
    """A Singapore brand → its ingredient(s)."""
    if not _HAS_SG:
        return None
    try:
        ings = sgr.brand_ingredients(entry["brand"])
    except Exception:  # noqa: BLE001
        ings = []
    if not ings:
        return None
    lex, _ns, _ = _lexicon()
    first = ings[0]
    base = lex.get(_norm(_SALT_WORDS_RE.sub("", first.lower())))
    query = base["query"] if base and base.get("kind") in ("library", "alias") else first
    return {"kind": "brand", "query": query, "display": entry["display"],
            "brand": entry["display"], "ingredients": ings}


# ---------------------------------------------------------------------------
# Drug classes & conditions (Drug Library / WHO ATC)
# ---------------------------------------------------------------------------
# User wording → substrings of class names in the library.
CLASS_ALIASES: List[Tuple[str, Tuple[str, ...]]] = [
    (r"statins?", ("hmg coa reductase inhibitor",)),
    (r"ssris?|selective serotonin reuptake inhibitors?", ("selective serotonin reuptake inhibitor",)),
    (r"snris?|serotonin(?: and)? norepinephrine reuptake inhibitors?",
     ("serotonin and norepinephrine reuptake inhibitor",)),
    (r"nsaids?|non ?steroidal anti ?inflammator(?:y|ies)(?: drugs?)?",
     ("nonsteroidal anti inflammatory drug",)),
    (r"benzos?|benzodiazepines?", ("benzodiazepine",)),
    (r"opioids?|opiates?|narcotics?", ("opioid agonist", "natural opium alkaloids")),
    (r"antibiotics?|antibacterials?", ("antibacterials for systemic use",)),
    (r"antihistamines?", ("antihistamines for systemic use", "histamine 1 receptor antagonist")),
    (r"beta ?blockers?|beta blocking agents?", ("beta blocking agents", "beta adrenergic blocker")),
    (r"ace inhibitors?", ("ace inhibitors", "angiotensin converting enzyme inhibitor")),
    (r"arbs?|angiotensin(?: ii| 2)? receptor blockers?",
     ("angiotensin ii receptor blockers", "angiotensin 2 receptor blocker")),
    (r"ppis?|proton pump inhibitors?", ("proton pump inhibitor",)),
    (r"h2 (?:blockers?|antagonists?|receptor antagonists?)", ("h2 receptor antagonist",
                                                              "histamine 2 receptor antagonist")),
    (r"calcium channel blockers?|ccbs?", ("calcium channel blocker",)),
    (r"diuretics?|water pills?", ("diuretic",)),
    (r"anticoagulants?|blood thinners?", ("anticoagulant",)),
    (r"antiplatelets?|antiplatelet (?:drugs|agents)", ("platelet aggregation inhibitor",)),
    (r"antidepressants?", ("antidepressants",)),
    (r"antipsychotics?|neuroleptics?", ("antipsychotics", "atypical antipsychotic")),
    (r"anticonvulsants?|antiepileptics?|anti[- ]?seizure (?:drugs|medications|meds)",
     ("antiepileptics", "anti epileptic agent")),
    (r"triptans?", ("selective serotonin 5ht1 agonists", "serotonin 1b and 1d receptor agonist")),
    (r"corticosteroids?|steroids?", ("corticosteroid",)),
    (r"bisphosphonates?", ("bisphosphonate",)),
    (r"sulfonylureas?|sulphonylureas?", ("sulfonylurea",)),
    (r"dpp ?4 inhibitors?|gliptins?", ("dipeptidyl peptidase 4 inhibitor",
                                       "dipeptidyl peptidase 4 dpp 4 inhibitors")),
    (r"glp ?1(?: receptor)? agonists?|glp ?1s?",
     ("glucagon like peptide 1 receptor agonist", "glucagon like peptide 1 glp 1 analogues")),
    (r"sglt ?2 inhibitors?|gliflozins?", ("sodium glucose cotransporter 2 inhibitor",
                                         "sodium glucose co transporter 2 sglt2 inhibitors")),
    (r"antivirals?", ("antivirals for systemic use",)),
    (r"antifungals?", ("antimycotics for systemic use", "azole antifungal")),
    (r"macrolides?", ("macrolides", "macrolide antimicrobial")),
    (r"cephalosporins?", ("cephalosporin",)),
    (r"fluoroquinolones?|quinolones?", ("fluoroquinolones", "quinolone antimicrobial")),
    (r"tetracyclines?", ("tetracyclines", "tetracycline class drug")),
    (r"penicillins?", ("penicillin class antibacterial", "penicillins")),
    (r"laxatives?", ("drugs for constipation", "laxative")),
    (r"antiemetics?|anti[- ]?nausea (?:drugs|medicines|meds)", ("antiemetics and antinauseants",)),
    (r"decongestants?", ("nasal decongestants", "sympathomimetics")),
    (r"expectorants?|mucolytics?", ("expectorants", "mucolytics")),
    (r"cough suppressants?|antitussives?|cough medicines?", ("cough suppressants",)),
    (r"muscle relaxants?", ("muscle relaxants",)),
    (r"stimulants?", ("centrally acting sympathomimetics", "central nervous system stimulant")),
    (r"sedatives?|sleeping pills?|hypnotics?|sleep aids?", ("hypnotics and sedatives",)),
    (r"local an(?:a)?esthetics?", ("anesthetics local", "local anesthetic")),
    (r"maois?|mao inhibitors?|monoamine oxidase inhibitors?", ("monoamine oxidase inhibitor",)),
    (r"tcas?|tricyclics?|tricyclic antidepressants?",
     ("non selective monoamine reuptake inhibitors", "tricyclic antidepressant")),
    (r"antihypertensives?|blood pressure (?:drugs|medications|medicines|meds|pills)",
     ("antihypertensives",)),
    (r"antimalarials?", ("antimalarials",)),
    (r"painkillers?|pain killers?|analgesics?", ("analgesics",)),
    (r"antacids?", ("antacids",)),
    (r"insulins?", ("insulins and analogues",)),
    (r"contraceptives?|birth control(?: pills?)?", ("hormonal contraceptives for systemic use",)),
    (r"vaccines?", ("vaccines",)),
]
_CLASS_ALIAS_RES = [(re.compile(r"\b(?:" + pat + r")\b"), subs) for pat, subs in CLASS_ALIASES]

# Condition → WHO ATC groups commonly used for it.
CONDITIONS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("cough", r"\bcough\w*", ("R05D", "R05C")),
    ("colds, flu & blocked nose", r"\b(?:colds?|flu|blocked nose|stuffy nose|congest\w*|"
     r"runny nose|sinus\w*)\b", ("R01B", "R01A", "R06A")),
    ("pain", r"\b(?:pain|aches?|headaches?|toothache|backache|period pain|cramps)\b",
     ("N02B", "M01A", "N02A")),
    ("fever", r"\bfevers?\b", ("N02B",)),
    ("migraine", r"\bmigraines?\b", ("N02C",)),
    ("allergies & hay fever", r"\b(?:allerg\w*|hay ?fever|itch\w*|hives|sneez\w*)\b",
     ("R06A", "R01AD")),
    ("heartburn, reflux & ulcers", r"\b(?:heartburn|reflux|gerd|indigestion|dyspepsia|"
     r"ulcers?|gastritis)\b", ("A02B", "A02A")),
    ("diarrhoea", r"\bdiarrh?o?ea\b", ("A07D", "A07B")),
    ("constipation", r"\bconstipat\w*", ("A06A",)),
    ("nausea, vomiting & motion sickness", r"\b(?:nause\w*|vomit\w*|motion sickness|"
     r"travel sickness)\b", ("A04A",)),
    ("insomnia", r"\b(?:insomnia|can'?t sleep|sleep(?:ing)? (?:problems?|trouble|aids?)|"
     r"sleepless\w*)\b", ("N05C",)),
    ("anxiety", r"\banxi\w*", ("N05B",)),
    ("depression", r"\bdepress\w*", ("N06A",)),
    ("high blood pressure", r"\b(?:high blood pressure|hypertension)\b",
     ("C09A", "C09C", "C08C", "C07A", "C03A")),
    ("high cholesterol", r"\b(?:cholesterol|hyperlipid\w*|dyslipid\w*)\b", ("C10A",)),
    ("diabetes", r"\b(?:diabet\w*|blood sugar|high sugar)\b", ("A10B", "A10A")),
    ("asthma & COPD", r"\b(?:asthma\w*|copd|wheez\w*|bronchospasm)\b", ("R03A", "R03B")),
    ("bacterial infections", r"\b(?:bacterial infections?|infections?|uti|urinary tract "
     r"infections?|strep throat|pneumonia)\b", ("J01",)),
    ("fungal infections", r"\b(?:fungal|thrush|ringworm|athlete'?s foot|yeast infections?)\b",
     ("D01A", "J02A")),
    ("viral infections (herpes, HIV, influenza)", r"\b(?:herpes|cold sores?|shingles|hiv|"
     r"influenza|covid\w*)\b", ("J05A",)),
    ("acne", r"\bacne\b", ("D10A", "D10B")),
    ("eczema, dermatitis & psoriasis", r"\b(?:eczema|dermatitis|psoriasis)\b", ("D07A", "D05A")),
    ("epilepsy & seizures", r"\b(?:epilep\w*|seizures?|convulsions?)\b", ("N03A",)),
    ("Parkinson's disease", r"\bparkinson\w*", ("N04B", "N04A")),
    ("gout", r"\bgout\b", ("M04A",)),
    ("osteoporosis", r"\bosteoporosis\b", ("M05B",)),
    ("erectile dysfunction", r"\b(?:erectile|impotence)\b", ("G04BE",)),
    ("quitting smoking", r"\b(?:quit(?:ting)? smoking|stop(?:ping)? smoking|smoking "
     r"cessation|nicotine addiction)\b", ("N07BA",)),
    ("weight loss", r"\b(?:obesity|weight loss|lose weight|losing weight|overweight)\b", ("A08A",)),
    ("malaria", r"\bmalaria\b", ("P01B",)),
    ("worms", r"\b(?:worms?|helminth\w*|deworm\w*)\b", ("P02",)),
    ("glaucoma", r"\bglaucoma\b", ("S01E",)),
    ("underactive thyroid", r"\b(?:hypothyroid\w*|underactive thyroid)\b", ("H03A",)),
    ("overactive thyroid", r"\b(?:hyperthyroid\w*|overactive thyroid)\b", ("H03B",)),
    ("contraception", r"\b(?:contracepti\w*|birth control|prevent pregnancy)\b", ("G03A",)),
    ("ADHD", r"\badhd\b|\battention deficit\b", ("N06BA",)),
    ("dementia", r"\b(?:alzheimer\w*|dementia)\b", ("N06D",)),
    ("psychosis, schizophrenia & bipolar", r"\b(?:schizophreni\w*|psychos[ie]s|psychotic|"
     r"bipolar)\b", ("N05A",)),
    ("angina", r"\bangina\b", ("C01D",)),
    ("blood clots", r"\b(?:blood clots?|thrombosis|dvt|atrial fibrillation)\b", ("B01A",)),
    ("muscle spasm", r"\b(?:muscle spasms?|spasticity)\b", ("M03B", "M03A")),
    ("arthritis & inflammation", r"\b(?:arthritis|inflammation|joint pain|rheumat\w*)\b", ("M01A",)),
    ("overactive bladder", r"\b(?:overactive bladder|incontinence|urinary urgency)\b", ("G04BD",)),
    ("enlarged prostate", r"\b(?:enlarged prostate|bph|prostat\w* hyperplasia)\b", ("G04CA",)),
]
_CONDITION_RES = [(label, re.compile(pat), codes) for label, pat, codes in CONDITIONS]


@lru_cache(maxsize=1)
def _class_index() -> List[Tuple[str, str]]:
    """[(normalised class name, slug)] — names cut at the first comma ('ACE
    inhibitors, plain' → 'ace inhibitors')."""
    if not _HAS_DL:
        return []
    out = []
    for slug, meta in dl.CLASSES.items():
        name = _norm((meta.get("name") or "").split(",")[0])
        if name:
            out.append((name, slug))
    return out


def find_classes(text: str) -> Tuple[List[str], str]:
    """Class slugs named in `text` (aliases first, then literal class names),
    most drugs first, plus the phrase that matched."""
    if not _HAS_DL:
        return [], ""
    t = _norm(text)
    padded = f" {t} "
    slugs: List[str] = []
    phrase = ""
    for rx, subs in _CLASS_ALIAS_RES:
        m = rx.search(t)
        if not m:
            continue
        phrase = phrase or m.group(0)
        for name, slug in _class_index():
            if any(s in name for s in subs) and slug not in slugs:
                slugs.append(slug)
    if not slugs:
        for name, slug in _class_index():
            if len(name) < 6 or name in COMMON_WORDS:
                continue
            if f" {name} " in padded or f" {name}s " in padded:
                slugs.append(slug)
                if len(name) > len(phrase):
                    phrase = name
    counts = dl.class_counts()

    def size(slug):
        if slug in counts:
            return counts[slug]
        return len(dl.drugs_in_class(slug))
    slugs.sort(key=lambda s: -size(s))
    return slugs[:6], phrase


def find_condition(text: str) -> Optional[Tuple[str, List[str]]]:
    """(condition label, ATC codes present in the library) for a condition named
    in `text`, or None."""
    t = (text or "").lower()
    for label, rx, codes in _CONDITION_RES:
        if rx.search(t):
            present = [c for c in codes if _HAS_DL and c in dl.CLASSES]
            if present:
                return label, present
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Chat shorthand → plain English, so the rules and the example matcher see
# 'does it make you sleepy' for 'does it make u sleepy'. Applied to the raw
# text before parsing (offsets are taken after this step).
_SLANG = [
    # lower-case only: 'U' after a number is units of insulin, 'IM' intramuscular
    (r"(?<!\d)(?<!\d )\bu\b", "you", 0), (r"\bim\b", "i'm", 0),
    (r"\bur\b", "your", re.I), (r"\bb4\b", "before", re.I), (r"\bhrs?\b", "hours", re.I),
    (r"\bmins\b", "minutes", re.I), (r"\bwhats\b", "what is", re.I), (r"\bwats\b", "what is", re.I),
    (r"\bwot\b", "what", re.I), (r"\bhow's\b", "how is", re.I), (r"\bpls\b|\bplz\b", "please", re.I),
    (r"\babt\b", "about", re.I), (r"\bw/o\b", "without", re.I), (r"\bw/(?=\s)", "with", re.I),
    (r"\bcuz\b|\bcoz\b", "because", re.I), (r"\bdont\b", "don't", re.I), (r"\bcant\b", "can't", re.I),
    (r"\bwont\b", "won't", re.I), (r"\bdoesnt\b", "doesn't", re.I), (r"\bisnt\b", "isn't", re.I),
    (r"\bive\b", "i've", re.I), (r"\bpreggo\b|\bpreggers\b", "pregnant", re.I),
    (r"\bmeds\b", "medicines", re.I), (r"\btabs\b", "tablets", re.I),
]
_SLANG_RES = [(re.compile(p, fl), r) for p, r, fl in _SLANG]


def expand_shorthand(raw: str) -> str:
    for rx, rep in _SLANG_RES:
        raw = rx.sub(rep, raw)
    return raw


def _normalize(raw: str) -> str:
    s = (raw.replace("’", "'").replace("‘", "'").replace("“", '"')
         .replace("”", '"').replace("–", "-").replace("—", "-"))
    low = s.lower()
    if len(low) != len(s):  # keep offsets aligned with the raw text
        low = "".join(c.lower() if len(c.lower()) == 1 else c for c in s)
    return low


def _looks_like_smiles(tok: str) -> bool:
    if Chem is None or len(tok) < 2 or len(tok) > 400:
        return False
    if tok.upper() in _ACRONYMS and tok.isalpha():
        return False
    if re.search(r"[()=#\[\]@/\\%]|\d", tok):
        pass
    elif not (tok.isalpha() and tok == tok.upper() and len(tok) >= 3):
        return False
    if re.fullmatch(r"[A-Za-z]+-?\d*", tok) and not tok.isupper():
        return False                        # 'B12', 'vitamin-d3'
    try:
        mol = Chem.MolFromSmiles(tok)
    except Exception:  # noqa: BLE001
        return False
    return mol is not None and mol.GetNumAtoms() >= 2


def _intent_matches(t2: str) -> List[Tuple[str, int, int, bool]]:
    found = []
    for name, strong, weak in _INTENT_RES:
        if strong is not None:
            for m in strong.finditer(t2):
                if m.end() > m.start():
                    found.append((name, m.start(), m.end(), True))
        if weak is not None:
            for m in weak.finditer(t2):
                if m.end() > m.start():
                    found.append((name, m.start(), m.end(), False))
    return found


def _select_intents(found: List[Tuple[str, int, int, bool]]) -> List[str]:
    by_intent: Dict[str, List[Tuple[int, int, bool]]] = {}
    for name, s, e, strong in found:
        by_intent.setdefault(name, []).append((s, e, strong))
    strong_any = {n for n, ms in by_intent.items() if any(st for _, _, st in ms)}
    keep: Dict[str, int] = {}
    for name, ms in by_intent.items():
        strong_ms = [(s, e) for s, e, st in ms if st]
        if not strong_ms:
            # weak-only: kept only when nothing else is strong
            if strong_any:
                continue
            strong_ms = [(s, e) for s, e, _ in ms]

        # a match fully inside a longer strong match of another intent is part
        # of that phrase ('how does drugx TREAT …' is mechanism, not uses)
        def subsumed(s, e):
            for other, oms in by_intent.items():
                if other == name:
                    continue
                for os_, oe, ost in oms:
                    if ost and os_ <= s and e <= oe and (oe - os_) > (e - s):
                        return True
            return False
        live = [(s, e) for s, e in strong_ms if not subsumed(s, e)]
        if live:
            keep[name] = min(s for s, _ in live)
    if "overview" in keep and len(keep) > 1:
        keep.pop("overview")
    return [n for n, _ in sorted(keep.items(), key=lambda kv: kv[1])]


@lru_cache(maxsize=4096)
def _is_intent_word(w: str) -> bool:
    """True when the word on its own is an intent keyword ('pharmacodynamics',
    'dosage', 'structure') — so it is never mistaken for a drug name."""
    for _name, strong, weak in _INTENT_RES:
        for rx in (strong, weak):
            if rx is None:
                continue
            m = rx.search(w)
            if m and m.group(0) == w:
                return True
    return False


def _focus_terms(t2: str, found) -> List[str]:
    chars = list(t2)
    for _name, s, e, _st in found:
        for i in range(s, e):
            chars[i] = " "
    rest = "".join(chars)
    out: List[str] = []
    for m in _TOKEN_RE.finditer(rest):
        w = m.group(0)
        if (w in STOPWORDS or w in FILLER or len(w) < 3 or w.isdigit()) and w not in LAY_SYNONYMS:
            continue
        if w in ("stop", "work", "drugx"):
            continue
        if w not in out:
            out.append(w)
    return out


def clip_query(raw: str) -> str:
    """Bound the query length, keeping both ends (the drug name is as likely to
    be at the end of a rambling question as at the start)."""
    raw = re.sub(r"\s+", " ", (raw or "")).strip()
    if len(raw) <= MAX_QUERY_LEN:
        return raw
    half = MAX_QUERY_LEN // 2 - 2
    return raw[:half] + " … " + raw[-half:]


# A relationship question: an explicit link word, or why/how + a verb of use.
_RELATION_RE = re.compile(
    r"\bantidotes?\b|\brevers(?:e|es|ed|ing|al)\b|"
    r"\bwhy\b.{0,80}\b(?:used|given|taken|prescribed|chosen|preferred|first[- ]line|helpful|"
    r"effective|good|works?|treats?|treating|cause[sd]?|combined|paired|added|together|"
    r"for|in|against|with)\b|"
    r"\bhow (?:does|do|can|is|are)\b.{0,80}\b(?:treats?|treating|help\w*|revers\w*|"
    r"work\w* (?:for|in|on|against)|cause[sd]?)\b", re.I)
# The thing on the other side of the link: everything after the link word.
_REL_TARGET_RE = re.compile(
    r"\b(?:antidotes? (?:for|to|in|against|of)|revers(?:e|es|ed|ing|al of)|"
    r"treat(?:s|ed|ing|ment of|ment for)?|(?:used|given|prescribed|chosen|preferred) "
    r"(?:for|in|to treat|against)|"
    r"(?:given|taken|used|combined|paired|added|mixed|prescribed|administered) "
    r"(?:together with|alongside|with|to|at|on|before|after|in the)|together with|alongside|"
    r"(?:give|gives|giving|use|uses|using|administer\w*) (?:\w+ ){0,2}(?:for|in|during|with|to treat)|"
    r"helps?(?: with| in| for)?|works? (?:for|in|on|against)|"
    r"effective (?:for|in|against)|good for|cause[sd]?|for|against)\s+(?P<target>.+)$", re.I)


def relation_target(raw: str, entities: List[Dict]) -> str:
    """'beta blocker overdose' for 'why is glucagon the antidote for beta blocker
    overdose?' — the other side of a drug↔X relationship question, or '' when
    the question isn't one. The first entity is the subject; its own name is
    removed from the target."""
    if not entities or not _RELATION_RE.search(raw):
        return ""
    m = _REL_TARGET_RE.search(raw)
    if not m:
        return ""
    target = m.group("target")
    subj = entities[0].get("text") or entities[0].get("display") or ""
    if subj:
        target = re.sub(re.escape(subj), " ", target, flags=re.I)
    target = re.sub(r"\s+", " ", target).strip(" ?!.,;:\"'")
    target = re.sub(r"^(?:a|an|the|patients? with|people with|someone with)\s+", "", target, flags=re.I)
    words = {w.lower() for w in re.findall(r"[a-z]+", target, re.I)}
    # 'how does it work in the brain' is a mechanism question
    if words and words <= _REL_BODY_WORDS:
        return ""
    # 'what's the antidote for <drug> overdose' is about the drug's OWN
    # overdose — but 'why give naloxone in an overdose' links two things
    own = subj and re.search(rf"\b{re.escape(subj)}\W+(?:overdose|overdosage|poisoning|toxicity)\b",
                             raw, re.I)
    if words and words <= _REL_BODY_WORDS | _REL_GENERIC_TARGETS and (
            own or not re.search(r"\b(?:why|how)\b", raw, re.I)):
        return ""
    return target if re.search(r"[a-z]{3}", target, re.I) else ""


# Words that never make a relationship target on their own.
_REL_BODY_WORDS = frozenset("""
the a an my your his her its their body brain heart liver kidney kidneys stomach gut blood
cells cell nerves nervous system cns muscles lungs skin bones human humans people person
""".split())
_REL_GENERIC_TARGETS = frozenset("overdose overdoses overdosage poisoning toxicity intoxication".split())


# 'does it raise/lower/affect …' (an effect question unless another topic is named)
_EFFECT_VERB_RE = re.compile(
    r"\b(?:does|do|can|will|could|would|might|may) (?:it|drugx|they) (?:raise|increase|lower|"
    r"decrease|reduce|elevate|worsen|affect|change|spike|drop|thin|damage|harm)\b")

# ---- allergies ----
_ALLERGY_RE = re.compile(r"\ballerg\w*")
# the allergen when it isn't a recognised drug: 'allergic to sulfa', 'nuts'
_ALLERGEN_WORD_RE = re.compile(
    r"\ballergic to (?:a |an |the )?([a-z][a-z\- ]{1,30}?)(?=\s+(?:and|but|so|can|could|is|will|"
    r"should|would|do|does|what|which)\b|[,.?!]|$)")


def _split_allergen(t: str, entities: List[Dict]) -> Optional[Dict]:
    """The entity that is the ALLERGEN in 'allergic to penicillin, can I take
    amoxicillin' / 'is amoxicillin safe with a penicillin allergy' — only when
    another drug is being asked about."""
    if len(entities) < 2:
        return None
    for en in entities:
        s, e = en["span"]
        before, after = t[max(0, s - 30):s], t[e:e + 14]
        if re.search(r"(?:allerg\w*|reactions?|sensitiv\w*|intoleran\w*) (?:to|with|from) (?:a |an )?$",
                     before) or re.match(r"\s*(?:allerg\w*|sensitivity|intolerance)", after):
            return en
    return None


# ---- nearest example in the question bank ----
USE_TEMPLATE_MATCHER = True
MATCH_STRONG = 0.86     # ≈ the same phrasing as a bank example: the bank decides
MATCH_WEAK = 0.55       # similar phrasing: used only when the rules found nothing
_MATCH_DROP = frozenset("a an the please some any".split())


def _match_features(text: str) -> List[str]:
    """Stemmed words + adjacent word pairs, articles dropped — question words
    ('how long', 'can i') are kept: they carry the type."""
    t = re.sub(r"\{(?:drug|drug2|brand)\}", " drugx ", text.lower())
    t = re.sub(r"\{\w+\}", " ", t)
    words = [stem(w) for w in _TOKEN_RE.findall(t) if w not in _MATCH_DROP]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


@lru_cache(maxsize=1)
def _template_index():
    """[(type, example, {feature: weight}, norm)] with IDF weights over all
    bank examples."""
    if _qb is None:
        return [], {}
    rows = [(qt, ex, set(_match_features(ex))) for qt in _qb.types() for ex in _qb.examples(qt)]
    n = len(rows) or 1
    df: Dict[str, int] = {}
    for _qt, _ex, feats in rows:
        for f in feats:
            df[f] = df.get(f, 0) + 1
    idf = {f: math.log((n + 1) / (c + 1)) + 1.0 for f, c in df.items()}
    index = []
    for qt, ex, feats in rows:
        vec = {f: idf[f] for f in feats}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        index.append((qt, ex, vec, norm))
    return index, idf


# Vocabulary match thresholds, set on held-out batch 1 (alone, the matcher
# picked the right topic for 92% of questions; at these settings it covers
# ~85% of them at ~98% accuracy) — and only used when the rules found nothing.
TYPE_MIN = 0.25         # minimum similarity to a question type…
TYPE_MARGIN = 0.04      # …and its lead over the best type with a different topic


@lru_cache(maxsize=1)
def _type_index():
    """One weighted vocabulary per answerable question type — all its examples
    pooled (sublinear term counts × IDF across types) — so a paraphrase is
    matched to the TYPE's wording, not to a single example sentence."""
    if _qb is None:
        return []
    docs = []
    for qt in _qb.types():
        exp = qt.get("expect") or {}
        if exp.get("mode") != "answer" or not exp.get("topics"):
            continue
        counts: Dict[str, int] = {}
        for ex in _qb.examples(qt):
            for f in set(_match_features(ex)):
                counts[f] = counts.get(f, 0) + 1
        docs.append((qt, counts))
    n = len(docs) or 1
    df: Dict[str, int] = {}
    for _qt, counts in docs:
        for f in counts:
            df[f] = df.get(f, 0) + 1
    idf = {f: math.log((n + 1) / (c + 1)) + 1.0 for f, c in df.items()}
    index = []
    for qt, counts in docs:
        vec = {f: (1.0 + math.log(c)) * idf[f] for f, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        index.append((qt, vec, norm, idf))
    return index


def type_match(text: str) -> Optional[Tuple[Dict, float, float]]:
    """(best question type, its similarity, the best similarity among types
    whose primary topic differs) for a question with drugs replaced by 'drugx'."""
    index = _type_index()
    if not index:
        return None
    idf = index[0][3]
    feats = set(_match_features(text)) - {"drugx"}
    q = {f: idf[f] for f in feats if f in idf}
    if not q:
        return None
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    scored = []
    for qt, vec, norm, _idf in index:
        dot = sum(w * vec[f] for f, w in q.items() if f in vec)
        scored.append((dot / (qn * norm), qt))
    scored.sort(key=lambda x: -x[0])
    best_s, best = scored[0]
    topic = best["expect"]["topics"][0]
    rival = next((s for s, qt in scored[1:] if qt["expect"]["topics"][0] != topic), 0.0)
    return best, best_s, rival


def template_match(text: str) -> Optional[Tuple[Dict, float, str]]:
    """(question type, cosine score, the example it matched) for the bank
    example nearest to `text` (drug names already replaced by 'drugx')."""
    index, idf = _template_index()
    if not index:
        return None
    feats = set(_match_features(text))
    q = {f: idf.get(f, math.log(len(index) + 1) + 1.0) for f in feats}
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    best = None
    for qt, ex, vec, norm in index:
        dot = sum(w * vec[f] for f, w in q.items() if f in vec)
        if not dot:
            continue
        score = dot / (qn * norm)
        if best is None or score > best[1]:
            best = (qt, score, ex)
    return best


def parse_query(raw: str, context: Optional[Dict] = None) -> Dict:
    """Parse one chat message.

    `context` is the previous turn: {"entity": {...}, "intents": [...]}.
    Returns a dict (stored in session state, so plain data only):
      mode       answer | compare | interaction | multi | class_list | condition |
                 clarify | unknown | help | thanks | empty
      entities   [{"kind", "query", "display", ...}]  — what to look up
      intents    ordered intent names (see INTENT_TITLES)
      focus      content words to rank sentences by ("coughing")
      notes      [str] — corrections/assumptions to tell the user
      suggestions [str] — did-you-mean names when nothing matched
    """
    raw = expand_shorthand(clip_query(raw))
    out: Dict = {"raw": raw, "mode": "answer", "entities": [], "weak": [], "intents": [],
                 "focus": [], "notes": [], "suggestions": [], "used_context": False,
                 "class_slugs": [], "class_phrase": "", "condition": None,
                 "pk_facts": [], "population": [], "interactant": "", "form": "",
                 "same_as": [], "allergen": "", "template": "", "template_score": 0.0}
    if not raw:
        out["mode"] = "empty"
        return out
    t = _normalize(raw)
    out["form"] = _qb.question_form(t) if _qb else ""
    # safety first: self-harm wording outranks everything (no doses shown)
    if _CRISIS_RE.search(t):
        out["mode"] = "crisis"
        return out
    if _GREETING_RE.match(t.strip()):
        out["mode"] = "help"
        return out
    if _THANKS_RE.match(t.strip()):
        out["mode"] = "thanks"
        return out
    if _HOW_TO_MAKE_RE.search(t) and _ILLICIT_RE.search(t):
        out["mode"] = "refuse"
        return out
    if _PET_RE.search(t) and _PET_CUE_RE.search(t):
        out["mode"] = "pets"
        out["urgent"] = bool(_PET_URGENT_RE.search(t))
        return out

    # ---- spans that can't be part of a drug name ----
    blocked = [False] * len(t)
    for m in _CLASS_MASK_RE.finditer(t):
        for i in range(m.start(), m.end()):
            blocked[i] = True

    entities: List[Dict] = []
    weak: List[Dict] = []

    # ---- SMILES ----
    for m in re.finditer(r"\S+", raw):
        tok = m.group(0).rstrip(",;:?!")
        if tok.endswith(".") and not _looks_like_smiles(tok):
            tok = tok[:-1]
        if _looks_like_smiles(tok):
            s, e = m.start(), m.start() + len(tok)
            entities.append({"kind": "smiles", "query": tok, "display": tok, "span": [s, e]})
            for i in range(s, e):
                blocked[i] = True

    # ---- lexicon n-grams (longest first) ----
    lex, nospace, _ = _lexicon()
    toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(t)]
    used = [False] * len(toks)
    for i, (_w, s, e) in enumerate(toks):
        if any(blocked[s:e]):
            used[i] = True
    i = 0
    while i < len(toks):
        if used[i]:
            i += 1
            continue
        hit = None
        for n in range(min(5, len(toks) - i), 0, -1):
            if any(used[i:i + n]):
                continue
            key = " ".join(w for w, _, _ in toks[i:i + n])
            entry = lex.get(key) or nospace.get(key.replace(" ", ""))
            if not entry:
                continue
            if n == 1 and (key in STOPWORDS or len(key) < 3):
                continue
            hit = (n, key, entry)
            break
        if not hit:
            i += 1
            continue
        n, key, entry = hit
        s, e = toks[i][1], toks[i + n - 1][2]
        ent = dict(entry)
        if ent["kind"] == "brand":
            ent = _brand_entity(entry)
            if not ent:
                i += 1
                continue
        ent.update(text=raw[s:e], span=[s, e])
        if n == 1 and (key in _WEAK_SUBSTANCES or _is_common(key)):
            weak.append(ent)
        else:
            entities.append(ent)
        for k in range(i, i + n):
            used[k] = True
        i += n

    # ---- typos (only words that aren't English / question words) ----
    for idx, (w, s, e) in enumerate(toks):
        if used[idx] or len(w) < 5 or not w.isalpha():
            continue
        if (_is_common(w) or w in STOPWORDS or w in FILLER or w in LAY_SYNONYMS
                or _is_intent_word(w)):
            continue
        cands = fuzzy_candidates(w)
        if not cands:
            continue
        best_key, dist = cands[0]
        tied = [k for k, d in cands if d == dist]
        if len(tied) == 1:
            entry = dict(lex[best_key])
            if entry["kind"] == "brand":
                entry = _brand_entity(entry) or {}
                if not entry:
                    continue
            entry.update(text=raw[s:e], span=[s, e], kind_matched=entry["kind"],
                         original=raw[s:e], distance=dist, kind="fuzzy",
                         corrected_display=entry["display"])
            # one-letter slip in a long name is certain; otherwise the app
            # tries the word as typed first (it may be a real, rarer chemical)
            entry["certain"] = dist == 1 and len(w) >= 7
            entities.append(entry)
            used[idx] = True
        else:
            out["suggestions"] += [lex[k]["display"] for k in tied if k in lex]

    entities.sort(key=lambda en: en["span"][0])
    # de-duplicate the same ingredient named twice ('paracetamol (acetaminophen)')
    seen_q: Dict[str, Dict] = {}
    uniq = []
    for en in entities:
        q = (en.get("query") or en.get("display") or "").lower()
        # a combination brand (Augmentin = amoxicillin + clavulanate) is NOT
        # the same medicine as its first ingredient
        if en.get("kind") == "brand" and len(en.get("ingredients") or []) > 1:
            q = "brand:" + (en.get("display") or q).lower()
        if q in seen_q:
            first = seen_q[q]
            # 'is paracetamol the same as acetaminophen' — two names, one drug
            if (en.get("display") or "").lower() != (first.get("display") or "").lower():
                out["same_as"] = [first.get("display") or q, en.get("display") or q,
                                  first.get("kind") or "", en.get("kind") or ""]
            continue
        seen_q[q] = en
        uniq.append(en)
    entities = uniq

    # ---- intents ----
    t2 = t
    for en in sorted(entities + (weak if not entities else []), key=lambda x: -x["span"][0]):
        s, e = en["span"]
        t2 = t2[:s] + "drugx" + t2[e:]
    found = _intent_matches(t2)
    intents = _select_intents(found)
    focus = _focus_terms(t2, found)

    out["form"] = _qb.question_form(t2) if _qb else ""
    # 'how long do the side effects last' asks about side effects, not PK (the
    # long 'how long do …' PK match swallows the side-effect words)
    if re.search(r"side[- ]?effects?[^?.!]{0,25}\b(?:last|go away|wear off|stop|disappear|"
                 r"fade|subside|get better|ease)|how long (?:does|do|will) (?:the |my |this )?"
                 r"(?!(?:one |a |each |single )?(?:dose|tablet|pill|capsule))(?:\w+ ){1,3}"
                 r"(?:from|on|caused by|of) drugx (?:last|go away|wear off)", t2):
        intents = ["side_effects"] + [x for x in intents if x not in ("side_effects", "pk")]
    # questions where one topic's words are decisive over incidental ones
    if re.search(r"drug tests?|urine tests?|false positives?|show (?:up )?(?:on|in) (?:a )?\w+ tests?", t):
        intents = ["pk"] + [x for x in intents if x != "pk"]
    if "dosing" in intents and re.search(
            r"\bcourse of|stay on (?:it|drugx)|be on (?:it|drugx)|for (?:years|months|life|a long time)|"
            r"how many (?:days|weeks) (?:should|do|must|can|will) (?:i|you)", t2):
        intents = ["dosing"] + [x for x in intents if x != "dosing"]
    if "overdose" in intents and re.search(r"overdos\w*|poison\w*|toxicity|too much|too many", t):
        intents = ["overdose"] + [x for x in intents if x != "overdose"]
    if "pregnancy" in intents and intents[0] in ("side_effects", "warnings"):
        intents = ["pregnancy"] + [x for x in intents if x != "pregnancy"]
    if re.search(r"(?:should|must) avoid|not suitable|unsuitable|contraindicat", t):
        intents = ["warnings"] + [x for x in intents if x != "warnings"]
    # storage words decide even when a longer 'how long does it last…' phrase
    # swallowed them
    if re.search(r"\b(?:fridge|refrigerat\w*|freezer|stor(?:e|ed|age|ing)|once opened|after opening|"
                 r"expir\w*|shelf[- ]?life|room temperature|out of date|use[- ]by)\b", t):
        intents = ["storage"] + [x for x in intents if x != "storage"]
    # 'does it cause dependence' is about dependence, not a side-effect list
    if "legal" in intents and re.search(r"\bcause\w* (?:physical |psychological )?"
                                        r"(?:dependence|addiction|withdrawal)", t):
        intents = ["legal"] + [x for x in intents if x != "legal"]
    # 'does it affect driving / birth control / the unborn baby' — the effect
    # verb alone doesn't make it a side-effect question when another topic is named
    if (len(intents) > 1 and intents[0] == "side_effects" and _EFFECT_VERB_RE.search(t2)
            and not re.search(r"side[- ]?effects?|adverse|\bcause", t2)):
        intents = intents[1:] + ["side_effects"]
    # decisive topics: a storage word ('can the syrup be kept out of the
    # fridge') or 'instead of' outranks incidental dosing/uses words
    for decisive in ("storage", "alternatives"):
        if decisive in intents:
            intents = [decisive] + [x for x in intents if x != decisive]

    # 'allergic to penicillin — can I take amoxicillin?': the allergen is not a
    # second subject; the question is about the other drug's warnings
    if _ALLERGY_RE.search(t):
        allergen = _split_allergen(t, entities)
        if allergen is not None:
            entities = [en for en in entities if en is not allergen]
            out["allergen"] = allergen.get("display") or allergen.get("query") or ""
        else:
            m = _ALLERGEN_WORD_RE.search(t)
            if m and not any(en["span"][0] <= m.start(1) < en["span"][1] for en in entities):
                out["allergen"] = m.group(1).strip()
        personal = (re.search(r"\ballergic\b|\ballergy to\b|\b\w+ allergy\b|\ballergic reaction",
                              t) and not re.search(r"\bfor (?:my |an? |the |seasonal )?allerg", t))
        if out["allergen"] or (entities and personal):
            intents = ["warnings"] + [x for x in intents if x != "warnings"]

    # 'is amoxicillin a (type of) penicillin' — the second name is a CLASS here
    if len(entities) >= 2 and re.search(
            r"\b(?:is|are) drugx (?:an? |one of the )?(?:(?:type|kind|form|sort|member|example|"
            r"class|part) of (?:the )?(?:an? )?)?drugx\b", t2):
        second = entities[1]
        if find_classes(second.get("text") or second.get("display") or "")[0]:
            entities = [entities[0]]
            intents = ["class"] + [x for x in intents if x != "class"]

    population = [p for p in ("children", "elderly") if p in intents]
    intents = [x for x in intents if x not in ("children", "elderly")]
    if population:
        out["population"] = population
        if not any(x in intents for x in ("dosing", "warnings", "side_effects", "pregnancy",
                                          "pk", "overdose")):
            intents.append("population")
    if "pk" in intents:
        out["pk_facts"] = [lbl for lbl, rx in _PK_FACT_RES if rx.search(t)]

    # phrasings the keyword rules miss: the nearest example in the question
    # bank decides (see question_bank.json — add a phrasing there to teach it)
    if USE_TEMPLATE_MATCHER:
        t_match = t2
        if "drugx" not in t_match and (context or {}).get("entity"):
            t_match = _PRONOUN_RE.sub("drugx", t_match)
        if "drugx" in t_match:
            # topics found only by WEAK cues ('can I take…' → warnings) are
            # guesses the example matcher may overrule
            strong_names = {n for n, _s, _e, st in found if st}
            weak_only = {n for n, _s, _e, _st in found} - strong_names
            specific = [x for x in intents if x not in ("overview", "search") and x not in weak_only]
            chosen = None
            hit = template_match(t_match)
            if hit and hit[1] >= MATCH_STRONG:
                qtype, score, _ex = hit
                exp = qtype.get("expect") or {}
                if exp.get("mode") == "answer" and exp.get("topics") and (
                        not specific or specific[0] not in exp["topics"]):
                    chosen = (qtype, score)
            elif not specific:
                tm = type_match(t_match)
                if tm and tm[1] >= TYPE_MIN and tm[1] - tm[2] >= TYPE_MARGIN:
                    chosen = (tm[0], tm[1])
            if chosen:
                qtype, score = chosen
                topic = qtype["expect"]["topics"][0]
                intents = [topic] + [x for x in intents if x not in (topic, "overview", "search")]
                vocab = {st for ex in _qb.examples(qtype) for st in tokens(ex)}
                focus = [f for f in focus if stem(f) not in vocab]
                if topic == "pk" and not out["pk_facts"]:
                    out["pk_facts"] = [lbl for lbl, rx in _PK_FACT_RES if rx.search(t)]
                out["template"], out["template_score"] = qtype["id"], round(score, 3)
                forms = qtype.get("form") or []
                if out["form"] in ("statement", "what") and forms and out["form"] not in forms:
                    out["form"] = forms[0]

    # ---- leftover words: an unrecognised drug/chemical name ----
    leftover_runs: List[Tuple[int, int]] = []
    run: Optional[List[int]] = None
    prev_end = -10
    for idx, (w, s, e) in enumerate(toks):
        ok = (not used[idx] and w not in STOPWORDS and not _is_common(w)
              and w not in FILLER and w not in LAY_SYNONYMS and len(w) >= 2
              and not _is_intent_word(w))
        if ok and run is not None and s - prev_end <= 2:
            run[1] = e
        elif ok:
            run = [s, e]
            leftover_runs.append(run)  # type: ignore[arg-type]
        else:
            run = None
        prev_end = e
    leftover_runs = [r for r in leftover_runs if not all(c.isdigit() for c in t[r[0]:r[1]])]

    # ---- decide ----
    ctx_entity = (context or {}).get("entity")
    ctx_intents = list((context or {}).get("intents") or [])
    has_pronoun = bool(_PRONOUN_RE.search(t))
    follow_on = bool(_FOLLOW_ON_RE.search(t.strip()))

    real_intents = [x for x in intents if x != "overview"]
    # A class name ('statins', 'benzodiazepines') is not an unknown chemical.
    cls_all, phrase_all = find_classes(t)
    if cls_all and phrase_all:
        ph = {stem(w) for w in _norm(phrase_all).split()}
        leftover_runs = [r for r in leftover_runs
                         if not {stem(w) for w in _norm(t[r[0]:r[1]]).split()} <= ph]

    def _free_entity(run) -> Dict:
        s, e = run
        # widen to the whole typed token so '2-methylpentane' keeps its locant
        while s > 0 and not raw[s - 1].isspace():
            s -= 1
        while e < len(raw) and not raw[e].isspace():
            e += 1
        text = re.sub(r"(?:'s|’s)$", "", raw[s:e].strip(" ?!.,;:\"'"))
        return {"kind": "free", "query": text, "display": text, "text": text, "span": [s, e]}

    if entities and leftover_runs and (_COMPARE_RE.search(t) or _CONNECT_RE.search(t)):
        # 'zyrtec vs claritin' — the second name isn't in the local lexicon,
        # but it is clearly the other thing being compared/combined
        extra = _free_entity(max(leftover_runs, key=lambda r: r[1] - r[0]))
        if extra["query"].lower() not in {(x.get("query") or "").lower() for x in entities}:
            entities.append(extra)
            entities.sort(key=lambda en: en["span"][0])
            focus = [f for f in focus if f not in _norm(extra["query"]).split()]
            t2 = t2.replace(_normalize(extra["text"]), "drugx", 1)
    if not entities:
        use_ctx = bool(ctx_entity) and (has_pronoun or follow_on)
        if weak and not leftover_runs:
            # 'is zinc good for colds' — the everyday substance IS the subject
            entities = weak[:1]
            weak = weak[1:]
        else:
            cls, phrase = cls_all, phrase_all
            cond = find_condition(t)
            if cond and _CONDITION_TRIGGER_RE.search(t) and not leftover_runs and not use_ctx:
                out.update(mode="condition", condition={"label": cond[0], "codes": cond[1]},
                           intents=intents, focus=focus)
                return out
            if cls and not leftover_runs and not use_ctx:
                out.update(mode="class_list", class_slugs=cls, class_phrase=phrase,
                           intents=intents, focus=focus)
                return out
            if leftover_runs and not use_ctx:
                ent = _free_entity(max(leftover_runs, key=lambda r: r[1] - r[0]))
                entities = [ent]
                focus = [f for f in focus if f not in _norm(ent["query"]).split()]
            elif ctx_entity and (real_intents or has_pronoun or follow_on):
                entities = [dict(ctx_entity)]
                out["used_context"] = True
            elif real_intents:
                out.update(mode="clarify", intents=intents, focus=focus)
                return out
            else:
                out.update(mode="unknown", focus=focus)
                return out
    reused = False
    if not intents and follow_on and ctx_intents and not out["used_context"]:
        # 'and morphine?' — the whole previous question, for the new drug
        intents = [x for x in ctx_intents if x != "search"] or ctx_intents
        out["notes"].append("same question as before")
        focus = focus or list((context or {}).get("focus") or [])
        for k in ("pk_facts", "population"):
            out[k] = out[k] or list((context or {}).get(k) or [])
        reused = True
    # weak substances next to a real drug are what it's taken *with*
    if weak and entities:
        out["interactant"] = weak[0]["display"].lower()
        if "interactions" not in intents and _CONNECT_RE.search(t):
            intents.insert(0, "interactions")
    if "interactions" in intents and not out["interactant"]:
        m = _INTERACTANT_RE.search(t)
        if m:
            out["interactant"] = m.group(0)
        elif reused:
            out["interactant"] = (context or {}).get("interactant") or ""

    # 'is amlodipine a calcium channel blocker?' — a class-membership question
    if re.search(r"\b(?:is|are) drugx (?:an?|one of the|considered an?|classed as an?|"
                 r"classified as an?|part of the)\b", t2):
        if cls_all and "class" not in intents:
            intents = ["class"] + [x for x in intents if x not in ("search", "overview")]
    cls, phrase = (cls_all, phrase_all) if "class" in intents else ([], "")
    out["class_slugs"], out["class_phrase"] = cls, phrase
    if cls:
        focus = [f for f in focus if f not in _norm(phrase).split()]

    strong = entities[:3]
    # 'why is glucagon the antidote for beta-blocker overdose' / 'how does
    # naloxone reverse morphine' — a question about the LINK between a drug and
    # something else, not about one topic of the drug
    rel = relation_target(raw, strong) if strong and not _COMPARE_RE.search(t2) else ""
    if rel:
        out.update(mode="relation", relation_target=rel, entities=strong[:1], weak=weak,
                   intents=intents or ["overview"], focus=focus)
        return out
    if len(strong) >= 2:
        others = [x for x in intents if x not in ("overview", "warnings")]
        if _COMPARE_RE.search(t2):
            out["mode"] = "compare"
            intents = intents or ["overview"]
        elif "interactions" in intents or (not others and _CONNECT_RE.search(t2)):
            out["mode"] = "interaction"
            intents = ["interactions"]
        else:
            out["mode"] = "multi"
            intents = intents or ["overview"]
    else:
        if not intents:
            intents = ["search"] if focus else ["overview"]
    out.update(entities=strong, weak=weak, intents=intents, focus=focus)
    for en in strong:
        if en.get("kind") == "fuzzy" and en.get("certain"):
            out["notes"].append(f"Interpreting “{en['original']}” as **{en['corrected_display']}**.")
    return out


# ---------------------------------------------------------------------------
# Sentence ranking (BM25 over light stems + lay→medical expansion)
# ---------------------------------------------------------------------------
_VERB_SUFFIXES = (("ingly", ""), ("edly", ""), ("ings", ""), ("ing", ""), ("ied", "y"),
                  ("ed", ""))


@lru_cache(maxsize=20000)
def stem(word: str) -> str:
    """Light suffix stripping so 'coughing'/'coughs'/'cough' and 'doses'/'dosing'
    /'dose' meet. Deliberately crude — it only has to be consistent."""
    w = word.lower()
    if len(w) <= 3:
        return w
    for suf, rep in _VERB_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: len(w) - len(suf)] + rep
            break
    else:
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif re.search(r"(?:ss|x|z|ch|sh)es$", w):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        elif w.endswith("ly") and len(w) > 5:
            w = w[:-2]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 4 and w[-1] == w[-2] and w[-1] not in "lsz":
        w = w[:-1]                    # 'stopped' → 'stopp' → 'stop'
    return w


def tokens(text: str) -> List[str]:
    return [stem(w) for w in _TOKEN_RE.findall((text or "").lower())
            if w not in STOPWORDS and len(w) > 1]


def expand_terms(focus: Sequence[str]) -> Dict[str, float]:
    """Focus words → {stem: weight}; lay words add their technical synonyms at a
    lower weight ('cough' → antitussive, cough reflex, medulla)."""
    weights: Dict[str, float] = {}
    for w in focus:
        for st in tokens(w):
            weights[st] = max(weights.get(st, 0.0), 1.0)
        for syn in LAY_SYNONYMS.get(w.lower(), ()):
            parts = tokens(syn)
            for st in parts:
                weights[st] = max(weights.get(st, 0.0), 0.6 if len(parts) == 1 else 0.45)
    return weights


def rank(docs: Sequence[str], weights: Dict[str, float], k1: float = 1.2,
         b: float = 0.75) -> List[Tuple[int, float]]:
    """BM25 scores [(doc index, score)] for docs matching ≥1 weighted term,
    best first."""
    if not docs or not weights:
        return []
    toks = [tokens(d) for d in docs]
    n = len(toks)
    avgdl = sum(len(t) for t in toks) / max(1, n) or 1.0
    df: Dict[str, int] = {}
    for tk in toks:
        for w in set(tk):
            if w in weights:
                df[w] = df.get(w, 0) + 1
    out = []
    for i, tk in enumerate(toks):
        if not tk:
            continue
        tf: Dict[str, int] = {}
        for w in tk:
            if w in weights:
                tf[w] = tf.get(w, 0) + 1
        if not tf:
            continue
        score = 0.0
        for w, f in tf.items():
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            score += weights[w] * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(tk) / avgdl))
        out.append((i, score))
    out.sort(key=lambda x: -x[1])
    return out


def matched_terms(text: str, weights: Dict[str, float]) -> List[str]:
    toks_ = set(tokens(text))
    return [w for w in weights if w in toks_]


if __name__ == "__main__":  # quick manual check
    import json
    import sys
    for q in sys.argv[1:] or ["how does dextromethorphan stop me from coughing",
                              "semaglutide pharmacodynamics", "structure of codeine"]:
        p = parse_query(q)
        print(q, "→", json.dumps({k: p[k] for k in ("mode", "intents", "focus", "notes",
                                                    "suggestions")}),
              [(e["kind"], e.get("query"), e.get("display")) for e in p["entities"]])
