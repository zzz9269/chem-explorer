EXCLUDE_HEADINGS = {
    "IUPAC Name", "InChI", "InChIKey", "SMILES", "Molecular Formula", "CAS",
    "Deprecated CAS", "European Community (EC) Number", "UNII", "UN Number",
    "ChEBI ID", "ChEMBL ID", "DrugBank ID", "DSSTox Substance ID", "HMDB ID",
    "ICSC Number", "KEGG ID", "NCI Thesaurus Code", "Nikkaji Number", "Wikidata",
    "Wikipedia", "MeSH Entry Terms", "Depositor-Supplied Synonyms", "Removed Synonyms",
    "Molecular Weight", "XLogP3", "Hydrogen Bond Donor Count",
    "Hydrogen Bond Acceptor Count", "Rotatable Bond Count", "Exact Mass",
    "Monoisotopic Mass", "Topological Polar Surface Area", "Heavy Atom Count",
    "Formal Charge", "Complexity", "Isotope Atom Count",
    "Defined Atom Stereocenter Count", "Undefined Atom Stereocenter Count",
    "Defined Bond Stereocenter Count", "Undefined Bond Stereocenter Count",
    "Covalently-Bonded Unit Count", "Compound Is Canonicalized",
    "Chemical Co-Occurrences in Literature", "Chemical-Gene Co-Occurrences in Literature",
    "Chemical-Disease Co-Occurrences in Literature", "Chemical-Organism Co-Occurrences in Literature",
    "Depositor-Supplied Patent Identifiers", "WIPO PATENTSCOPE", "Chemical Co-Occurrences in Patents",
    "Chemical-Disease Co-Occurrences in Patents", "Chemical-Gene Co-Occurrences in Patents",
    "Chemical-Organism Co-Occurrences in Patents", "MeSH Tree", "NCI Thesaurus Tree",
    "ChEBI Ontology", "KEGG: EDC", "KEGG: Pesticides", "ChemIDplus", "CAMEO Chemicals",
    "ChEMBL Target Tree", "UN GHS Classification", "EPA CPDat Classification",
    "NORMAN Suspect List Exchange Classification", "EPA DSSTox Classification",
    "EPA Substance Registry Services Tree", "CCS Classification - Baker Lab",
    "Chemicals in PubChem from Regulatory Sources",
    "ATCvet Classification", "PubChem CID", "Dates", "3D Conformer", "CCDC Number",
    "COD Number", "Associated Article", "AIDS Number", "NSC Number", "PharmGKB ID",
    "Pharos Ligand ID", "RXCUI", "Related Compounds", "PubChem Reference Collection SID",
    "Related Substances", "Entrez Crosslinks",
    "PDBe Ligand Code", "PDBe Structure Code", "KEGG: Drug", "KEGG: ATC",
    "KEGG: Target-based Classification of Drugs", "KEGG: JP15",
    "KEGG: Risk Category of Japanese OTC Drugs", "KEGG: OTC drugs", "KEGG: Animal Drugs",
    "KEGG: Drug Groups", "WHO ATC Classification System", "IUPHAR/BPS Guide to PHARMACOLOGY Target Classification",
    "CCSBase Classification", "EPA TSCA and CDR Classification", "LOTUS Tree", "DEA Code Number",
    "Dates", "KEGG: Metabolite", "KEGG: Lipid", "Lipid Maps ID (LM_ID)",
    "Crystal Structure Data",
    "Metabolomics Workbench ID", "KEGG: USP", "KEGG: Drug Classes", "MolGenie Organic Chemistry Ontology",
    
}



# grouping_config.py
# Keyword groups for grouping PubChem headings by heuristic matching.
# Keep keywords short and lowercase. Add variants (hyphen/space) where appropriate.

GROUP_KEYWORDS = {
    # Physical & experimental properties (include many variants & abbreviations)
    "Physical Properties": [
        "boiling", "boiling point", "melting", "melting point", "flash point", "flashpoint",
        "solub", "solubility", "logs", "log s", "log s.", "logsol", "logsw", "logp", "log d", "logd",
        "vapor pressure", "vapor", "vapor density", "density", "viscosity", "heat of combustion",
        "heat of vaporization", "surface tension", "refractive", "ionization potential",
        "critical temperature", "critical pressure", "autoignit", "autoignition",
        "stability", "shelf life", "odor threshold", "kovats", "kovats retention", "henry",
        "henry's law", "henry's law constant", "pka", "dissociation", "ph"
    ],

    # Structures & conformers
    "Structures & Conformers": [
        "structure", "2d", "2d structure", "3d", "3d conformer", "conformer", "crystal", "ccdc",
        "cod number", "molblock", "pdb", "coordinates", "coordinates (x,y,z)"
    ],

    # Names & identifiers
    "Names & Identifiers": [
        "name", "names", "synonym", "synonyms", "cas", "cas number", "pubchem cid", "cid",
        "rtecs", "jecfa", "unii", "inchikey", "inchi", "smiles", "molecular formula", "formula",
        "iupac", "other identifiers", "fema number", "rxnorm", "reactive group"
    ],

    # Computed / RDKit etc.
    "Computed Properties / Descriptors": [
        "computed", "descriptor", "computed descriptors", "molecular weight", "exact mass",
        "monoisotopic", "tpsa", "heavy atom", "rotatable", "complexity", "formal charge",
        "clogp", "xlogp", "mol_weight", "mol weight"
    ],

    # Experimental Properties & Measurements (lab-observed)
    "Experimental Properties & Measurements": [
        "experimental", "experimental properties", "physical description", "color", "color/form",
        "odor", "odour", "taste", "other experimental", "springermaterials",
        "collision cross", "reactivity profile", "dissociation", "reactivity alerts"
    ],

    # Safety & Hazards (fire, spillage, handling, PPE, first-aid)
    "Safety & Hazards": [
        "hazard", "hazards", "hazards identification", "primary hazards", "hazard summary",
        "fire", "fire hazards", "firefighting", "firefighting hazards", "fire potential",
        "flammable", "flammability", "flame", "toxic combustion", "toxic combustion products",
        "first aid", "first aid measures", "spillage", "accidental release", "cleanup",
        "disposal", "disposal methods", "prevent", "handling", "storage", "ppe", "personal protective",
        "skin prevention", "eye prevention", "ingestion prevention", "preventive measures",
        "hazardous reactivities", "incompatibilities", "explosive", "nonfire spill response", "isolation and evacuation", "physical dangers", "immediately dangerous to life or health (idlh)", "csl reaction information", "corrosivity"
    ],

    # Exposure & Occupational (OSHA/NIOSH, REL, PEL, exposure control)
    "Exposure & Occupational": [
        "osha", "niosh", "rel", "pel", "tlv", "oel", "exposure control", "exposure prevention",
        "respirator", "breathing", "inhalation", "inhalation risk", "effects of short term",
        "effects of long term", "inhalation prevention", "medical surveillance",
        "probable routes of human exposure", "exposure routes", "biological half-life",
        "minimum risk level", "mrl", "aecl", "aege", "aegl", "aehl", "skin, eye, and respiratory irritations", "other standards regulations and guidelines", "aegls table", "aegls notes", "cercla reportable quantities"
    ],

    # Regulatory, Transport & Legal
    "Regulatory & Transport": [
        "regulatory", "transport", "dot", "un number", "un classification", "shipping",
        "standard transportation", "shipment", "fifra", "fda", "fda requirements", "epa", "tsca",
        "guide", "emergency planning", "emergency response planning", "dot emergency",
        "dot id", "dot id and guide", "shipping name", "standard transportation number"
    ],

    # Toxicity & Health
    "Toxicity & Health": [
        "toxicity", "toxicological", "toxicity summary", "reported fatal dose", "carcinogen",
        "hepatotoxicity", "evidence for carcinogenicity", "carcinogen classification",
        "health effects", "signs and symptoms", "target organs", "treatment", "antidote",
        "antidote and emergency treatment", "adverse effects", "acute effects", "health hazards",
        "effects during pregnancy", "effects during pregnancy and lactation",
        "associated disorders", "drug induced liver injury", "livertox summary", "skin, eye, and respiratory irritations", "allowable tolerances"
    ],

    # Environmental & Fate
    "Environmental & Fate": [
        "environment", "ecotoxic", "ecotoxicity", "environmental", "biodegrad", "bioconcentration",
        "soil", "water", "volatiliz", "effluent", "atmospheric", "natural pollution", "artificial pollution",
        "environmental fate", "environmental biodegradation", "abiotic degradation", "body burden",
        "average daily intake", "food survey", "plant concentrations", "fish", "seafood", "milk concentrations", "decomposition", "rcra requirements"
    ],

    # Spectra & Analytical
    "Spectra & Analytical": [
        "nmr", "1d nmr", "1h nmr", "13c", "13c nmr", "mass spectrom", "mass spectrometry",
        "gc-ms", "gc ms", "ms-ms", "ms ms", "uv spectra", "lc-ms", "lc ms",
        "ir spectra", "ftir", "raman", "spectra", "kovats", "kovats retention", "analysis", "analytical",
        "sampling procedures", "analytical methods", "identification", "clinical laboratory", "niosh analytical",
        "lc-ms/ms", "lc ms/ms", "ionization efficiency"
    ],

    # Classification, Standards & Registries
    "Classification, Standards & Registries": [
        "classification", "registry", "jecfa", "who", "ifra", "efsa", "epa safer choice",
        "pesticides", "food additive", "food additives", "fcs", "gras", "endocrine",
        "chemical classes", "consumer product information", "dea drug and chemical information", "dea listed chemicals"
    ],

    # Manufacturing, Uses & Commerce
    "Manufacturing, Uses & Commerce": [
        "use", "uses", "fema flavor profile", "manufactur", "formulation", "formulations", "preparation", "preparations",
        "consumption", "production", "imports", "exports", "u.s. imports", "u.s. exports",
        "household products", "industry uses", "consumer uses", "impurities", "general manufacturing"
    ],

    # Pharmacology & Clinical (drugs, PK/PD, clinical use)
    "Pharmacology & Clinical": [
        "pharmacolog", "pharmacodynamic", "pharmacodynamics", "pharmacokinetic",
        "mechanism of action", "absorption", "distribution and excretion",
        "metabolism", "metabolites", "biological half-life", "therapeutic",
        "drug indication", "drug and medication", "drug classes", "drug label",
        "atc code", "human drugs", "animal drugs", "clinical trial",
        "tissue location", "cellular location", "max phase", "first approval",
        "black box warning", "route of administration", "availability type",
        "drugcentral", "caco2", "permeability"
    ],

    # Interactions, Pathways & Bioactivity
    "Interactions, Pathways & Bioactivity": [
        "interact", "interaction", "interactions", "pathway", "pathways", "bioassay", "ligand", "protein", "metabolite pathways",
        "biochemical reactions", "transformations", "mechanism of action", "drug-food", "drug-food interactions", "lipids"
    ],

    # Literature & References
    "Literature & References": [
        "literature", "reference", "pubmed", "consolidated references", "special reports",
        "history and incidents", "disease and references", "springer", "thieme", "wiley", "nature"
    ],

    # Related Records & Vendors
    "Related Records & Vendors": [
        "related", "related compounds", "substances", "vendors", "pubchem reference", "entrez", "linkout",
        "household products database", "chemical vendors", "substances by category", "other relationships", "associated chemicals"
    ],

    # Toxicology Programs & Studies
    "Toxicology Programs & Studies": [
        "national toxicology", "ntp", "toxcast", "tox21", "ongoing test", "populations at special risk",
        "nci", "studies", "testing status"
    ],

    # Patents (kept small)
    "Patents": [
        "patent", "patents", "patentscope", "wipo", "depositor-supplied patent", "patent identifiers"
    ],

    # fallback group - keep but should rarely be used now
    "Other": []
}


# ---------------- Exact (normalized) mappings for literal PubChem headings -----------
# Normalization in your code: (heading or "").strip().lower()
# Map long/exact headings -> group names (guaranteed placement).
# I included all non-excluded headings from your big "Ethanol" header list above.
CUSTOM_GROUP_MAP = {
    # Structures & descriptors
    "structures": "Structures & Conformers",
    "2d structure": "Structures & Conformers",
    "3d conformer": "Structures & Conformers",        # **if you keep 3d conformer** but your EXCLUDE may remove it
    "crystal structures": "Structures & Conformers",
    "cod number": "Structures & Conformers",

    # Names & identifiers
    "names and identifiers": "Names & Identifiers",
    "other identifiers": "Names & Identifiers",
    "fema number": "Names & Identifiers",
    "rtecs number": "Classification, Standards & Registries",
    "jecfa number": "Classification, Standards & Registries",
    "rxnorm": "Names & Identifiers",

    # general record text
    "record description": "Literature & References",
    "computed descriptors": "Computed Properties / Descriptors",
    "computed properties": "Computed Properties / Descriptors",
    "molecular weight": "Computed Properties / Descriptors",

    # experimental / physical
    "chemical and physical properties": "Physical Properties",
    "experimental properties": "Experimental Properties & Measurements",
    "physical description": "Experimental Properties & Measurements",
    "color/form": "Experimental Properties & Measurements",
    "color": "Experimental Properties & Measurements",
    "odor": "Experimental Properties & Measurements",
    "taste": "Experimental Properties & Measurements",
    "boiling point": "Physical Properties",
    "melting point": "Physical Properties",
    "flash point": "Physical Properties",
    "solubility": "Physical Properties",
    "density": "Physical Properties",
    "vapor density": "Physical Properties",
    "vapor pressure": "Physical Properties",
    "logp": "Physical Properties",
    "henry's law constant": "Physical Properties",
    "stability/shelf life": "Physical Properties",
    "autoignition temperature": "Physical Properties",
    "viscosity": "Physical Properties",
    "heat of combustion": "Physical Properties",
    "heat of vaporization": "Physical Properties",
    "surface tension": "Physical Properties",
    "ionization potential": "Physical Properties",
    "odor threshold": "Experimental Properties & Measurements",
    "refractive index": "Physical Properties",
    "dissociation constants": "Physical Properties",
    "kovats retention index": "Spectra & Analytical",
    "other experimental properties": "Experimental Properties & Measurements",
    "springermaterials properties": "Physical Properties",

    # classification / uses
    "chemical classes": "Classification, Standards & Registries",
    "drugs": "Pharmacology & Clinical",
    "human drugs": "Pharmacology & Clinical",
    "animal drugs": "Pharmacology & Clinical",
    "cosmetics": "Classification, Standards & Registries",
    "endocrine disruptors": "Classification, Standards & Registries",
    "flavoring agents": "Classification, Standards & Registries",
    "food additives": "Classification, Standards & Registries",
    "food contact substances": "Classification, Standards & Registries",
    "fragrances": "Classification, Standards & Registries",
    "pesticides": "Classification, Standards & Registries",
    "solvents": "Manufacturing, Uses & Commerce",

    # spectra
    "spectral information": "Spectra & Analytical",
    "1d nmr spectra": "Spectra & Analytical",
    "1h nmr spectra": "Spectra & Analytical",
    "13c nmr spectra": "Spectra & Analytical",
    "17o nmr spectra": "Spectra & Analytical",
    "2d nmr spectra": "Spectra & Analytical",
    "1h-1h nmr spectra": "Spectra & Analytical",
    "1h-13c nmr spectra": "Spectra & Analytical",
    "mass spectrometry": "Spectra & Analytical",
    "gc-ms": "Spectra & Analytical",
    "ms-ms": "Spectra & Analytical",
    "other ms": "Spectra & Analytical",
    "uv spectra": "Spectra & Analytical",
    "ir spectra": "Spectra & Analytical",
    "ftir spectra": "Spectra & Analytical",
    "atr-ir spectra": "Spectra & Analytical",
    "vapor phase ir spectra": "Spectra & Analytical",
    "raman spectra": "Spectra & Analytical",
    "other spectra": "Spectra & Analytical",

    # related records / vendors
    "related records": "Related Records & Vendors",
    "related compounds with annotation": "Related Records & Vendors",
    "substances": "Related Records & Vendors",
    "pubchem reference collection sid": "Related Records & Vendors",
    "substances by category": "Related Records & Vendors",
    "entrez crosslinks": "Related Records & Vendors",
    "ncbi linkout": "Related Records & Vendors",
    "chemical vendors": "Related Records & Vendors",
    "household products": "Manufacturing, Uses & Commerce",
    "household products database tree": "Related Records & Vendors",

    # pharmacology / clinical
    "drug and medication information": "Pharmacology & Clinical",
    "drug indication": "Pharmacology & Clinical",
    "drug classes": "Pharmacology & Clinical",
    "who essential medicines": "Classification, Standards & Registries",
    "fda approved drugs": "Regulatory & Transport",
    "fda orange book": "Regulatory & Transport",
    "fda national drug code directory": "Regulatory & Transport",
    "drug labels": "Pharmacology & Clinical",
    "clinical trials": "Pharmacology & Clinical",
    "clinicaltrials.gov": "Pharmacology & Clinical",
    "eu clinical trials register": "Pharmacology & Clinical",
    "niph clinical trials search of japan": "Pharmacology & Clinical",
    "ema drug information": "Pharmacology & Clinical",
    "therapeutic uses": "Pharmacology & Clinical",
    "drug warnings": "Toxicity & Health",
    "reported fatal dose": "Toxicity & Health",
    "drug tolerance": "Toxicity & Health",

    # food / standards
    "food additive classes": "Classification, Standards & Registries",
    "fda substances added to food": "Classification, Standards & Registries",
    "fda generally recognized as safe - gras notices": "Classification, Standards & Registries",
    "fda food contact substances (fcs)": "Classification, Standards & Registries",
    "associated foods": "Manufacturing, Uses & Commerce",
    "evaluations of the joint fao/who expert committee on food additives - jecfa": "Classification, Standards & Registries",

    # agro / pesticide
    "agrochemical category": "Classification, Standards & Registries",
    "eu pesticides data": "Classification, Standards & Registries",

    # pharmacology / biochem detail
    "pharmacology and biochemistry": "Pharmacology & Clinical",
    "pharmacodynamics": "Pharmacology & Clinical",
    "mesh pharmacological classification": "Pharmacology & Clinical",
    "fda pharmacological classification": "Pharmacology & Clinical",
    "atc code": "Pharmacology & Clinical",

    # absorption / metabolism
    "absorption, distribution and excretion": "Pharmacology & Clinical",
    "metabolism/metabolites": "Pharmacology & Clinical",
    "biological half-life": "Pharmacology & Clinical",
    "mechanism of action": "Pharmacology & Clinical",
    "human metabolite information": "Pharmacology & Clinical",
    "tissue locations": "Pharmacology & Clinical",
    "cellular locations": "Pharmacology & Clinical",
    "metabolite pathways": "Interactions, Pathways & Bioactivity",
    "biochemical reactions": "Interactions, Pathways & Bioactivity",
    "transformations": "Interactions, Pathways & Bioactivity",

    # drug / ADMET metadata (previously fell into "Other")
    "drugcentral": "Pharmacology & Clinical",
    "caco2 permeability": "Pharmacology & Clinical",
    "max phase": "Pharmacology & Clinical",
    "first approval": "Pharmacology & Clinical",
    "black box warning": "Pharmacology & Clinical",
    "availability type": "Pharmacology & Clinical",
    "route of administration": "Pharmacology & Clinical",
    "pfas and fluorinated organic compounds in pubchem": "Classification, Standards & Registries",

    # manufacturing / commerce
    "use and manufacturing": "Manufacturing, Uses & Commerce",
    "uses": "Manufacturing, Uses & Commerce",
    "use classification": "Manufacturing, Uses & Commerce",
    "industry uses": "Manufacturing, Uses & Commerce",
    "consumer uses": "Manufacturing, Uses & Commerce",
    "methods of manufacturing": "Manufacturing, Uses & Commerce",
    "impurities": "Manufacturing, Uses & Commerce",
    "formulations/preparations": "Manufacturing, Uses & Commerce",
    "consumption patterns": "Manufacturing, Uses & Commerce",
    "u.s. production": "Manufacturing, Uses & Commerce",
    "u.s. imports": "Manufacturing, Uses & Commerce",
    "u.s. exports": "Manufacturing, Uses & Commerce",
    "general manufacturing information": "Manufacturing, Uses & Commerce",

    # analytic / sampling
    "sampling procedures": "Spectra & Analytical",
    "identification": "Spectra & Analytical",
    "analytic laboratory methods": "Spectra & Analytical",
    "clinical laboratory methods": "Spectra & Analytical",
    "niosh analytical methods": "Spectra & Analytical",

    # safety/transport/regulatory specifics
    "hazards identification": "Safety & Hazards",
    "gHS classification": "Safety & Hazards",  # normalized may be "ghs classification"
    "nfpa hazard classification": "Safety & Hazards",
    "dot hazard classification": "Regulatory & Transport",
    "transport information": "Regulatory & Transport",
    "dot emergency guidelines": "Regulatory & Transport",
    "dot id and guide": "Regulatory & Transport",
    "shipping name/ number dot/un/na/imo": "Regulatory & Transport",
    "standard transportation number": "Regulatory & Transport",
    "shipment methods and regulations": "Regulatory & Transport",
    "dot label": "Regulatory & Transport",
    "un classification": "Regulatory & Transport",
    "fifra requirements": "Regulatory & Transport",
    "fda requirements": "Regulatory & Transport",
    "other safety information": "Safety & Hazards",
    "toxic combustion products": "Safety & Hazards",

    # toxicity & evidence
    "toxicity": "Toxicity & Health",
    "toxicological information": "Toxicity & Health",
    "toxicity summary": "Toxicity & Health",
    "evidence for carcinogenicity": "Toxicity & Health",
    "carcinogen classification": "Toxicity & Health",
    "health effects": "Toxicity & Health",
    "effects during pregnancy and lactation": "Toxicity & Health",
    "exposure routes": "Exposure & Occupational",
    "signs and symptoms": "Toxicity & Health",
    "target organs": "Toxicity & Health",
    "treatment": "Toxicity & Health",
    "antidote and emergency treatment": "Toxicity & Health",
    "medical surveillance": "Exposure & Occupational",

    # ongoing studies / programs
    "ongoing test status": "Toxicology Programs & Studies",
    "national toxicology program studies": "Toxicology Programs & Studies",
    "populations at special risk": "Toxicology Programs & Studies",

    # ecological / environmental
    "ecological information": "Environmental & Fate",
    "epa ecotoxicity": "Environmental & Fate",
    "ecotoxicity values": "Environmental & Fate",
    "ecotoxicity excerpts": "Environmental & Fate",
    "icsc environmental data": "Environmental & Fate",
    "environmental fate/exposure summary": "Environmental & Fate",
    "natural pollution sources": "Environmental & Fate",
    "artificial pollution sources": "Environmental & Fate",
    "environmental fate": "Environmental & Fate",
    "environmental biodegradation": "Environmental & Fate",
    "environmental abiotic degradation": "Environmental & Fate",
    "environmental bioconcentration": "Environmental & Fate",
    "soil adsorption/mobility": "Environmental & Fate",
    "volatilization from water/soil": "Environmental & Fate",
    "environmental water concentrations": "Environmental & Fate",
    "effluent concentrations": "Environmental & Fate",
    "atmospheric concentrations": "Environmental & Fate",
    "food survey values": "Environmental & Fate",
    "plant concentrations": "Environmental & Fate",
    "other environmental concentrations": "Environmental & Fate",
    "probable routes of human exposure": "Exposure & Occupational",
    "body burden": "Environmental & Fate",
    "efsa openfoodtox": "Classification, Standards & Registries",
    "efsa outputs": "Classification, Standards & Registries",

    # literature / references
    "literature": "Literature & References",
    "consolidated references": "Literature & References",
    "nlm curated pubmed citations": "Literature & References",
    "springer nature references": "Literature & References",
    "thieme references": "Literature & References",
    "wiley references": "Literature & References",
    "nature journal references": "Literature & References",
    "chemical co-occurrences in literature": "Literature & References",

    # misc / history / incidents
    "history and incidents": "Literature & References",
    "special reports": "Literature & References",
    "associated disorders and diseases": "Toxicity & Health",
    "disease and references": "Literature & References",

    # classification/registries extras
    "epa safer choice": "Classification, Standards & Registries",
    "chemicals in pubchem from regulatory sources": "Classification, Standards & Registries",
    "household products database tree": "Related Records & Vendors",
    "consumer product information database classification": "Classification, Standards & Registries",
    "fda drug type and pharmacologic classification": "Classification, Standards & Registries"
}

# ---------------- Helper (optional) - print leftovers in "Other" ----------------
# After your grouping loop, add this check to help catch anything still ungrouped:
# leftover = grouped.get("Other", {})
# if leftover:
#     st.warning(f"Ungrouped headings (Other): {len(leftover)}")
#     for h in sorted(leftover.keys()):
#         st.text(f"- {h}")
#     # optionally print them to console for copy/paste into CUSTOM_GROUP_MAP
