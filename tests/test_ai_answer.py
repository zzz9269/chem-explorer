"""The Claude Haiku layer, with no API calls: the number checks, the conversation loop
against a scripted client, spend limits, and what the app shows or falls back to.
tests/eval_ai_questions.py is the live run that spends credit."""
import json
import os
from types import SimpleNamespace as NS

import pytest
from streamlit.testing.v1 import AppTest

import ai_answer as ai

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chem_v4.py")
# parses as an unknown name, so the app fetches nothing before calling the model
NO_LOOKUP_Q = "what's the capital of france"


def _ev(*texts, question=""):
    ev = ai.Evidence(question)
    for t in texts:
        ev.add(t, "FDA label", "https://example.org")
    return ev


def _raw(headline="**No.**", sources=(), **extra):
    return dict({"kind": "answer", "headline": headline, "headline_sources": list(sources), "safety": [],
                 "points": [], "chain": [], "chain_sources": [], "table_columns": [], "table_rows": [],
                 "followups": []}, **extra)


# -------------------------------------------------------------- number checks
@pytest.mark.parametrize("text", ["Take 500 mg", "every 8 hours", "dose 500", "call 200 mg"])
def test_a_number_from_nowhere_is_rejected(text):
    assert ai.claim_ok(text, ["S1"], _ev("Take as directed by your doctor."))[0] is False


def test_a_number_in_the_cited_snippet_passes():
    assert ai.claim_ok("Take 500 mg", ["S1"], _ev("The dose is 500 mg twice daily."))[0] is True


def test_a_number_must_be_in_what_it_cites_not_just_any_snippet():
    ev = _ev("The dose is 500 mg.", "Half-life is 12 hours.")
    assert ai.claim_ok("Half-life is 500 hours", ["S2"], ev)[0] is False


def test_thousands_are_written_either_way():
    ev = _ev("Do not exceed 4000 mg in 24 hours.")
    assert ai.claim_ok("Maximum 4,000 mg in 24 hours", ["S1"], ev)[0] is True


@pytest.mark.parametrize("text", ["blocks H1 receptors", "inhibits CYP3A4", "type 2 diabetes",
                                  "a phase 3 trial", "5-HT reuptake", "D2 antagonist"])
def test_numbers_that_name_things_arent_measures(text):
    assert ai.claim_ok(text, [], _ev())[0] is True


def test_an_answer_with_no_source_may_not_state_a_measure():
    assert ai.claim_ok("The usual dose is 500 mg", [], _ev())[0] is False


def test_the_readers_own_number_and_the_emergency_number_need_no_source():
    ev = _ev("Quick medical attention is critical.", question="my toddler swallowed 10 panadol")
    assert ai.claim_ok("**Call 995 now**: 10 tablets is an emergency.", ["S1"], ev)[0] is True


def test_an_unknown_source_id_counts_as_no_source():
    ok, ids = ai.claim_ok("Take 500 mg", ["S9"], _ev("The dose is 500 mg."))
    assert ok is False and ids == []


# ---------------------------------------------------------------- verify
def test_a_headline_that_fails_fails_the_answer():
    assert ai.verify(_raw("**Take 900 mg**", ["S1"]), _ev("The dose is 500 mg.")) is None


def test_a_side_point_that_fails_is_only_dropped():
    raw = _raw("**Yes.**", ["S1"], points=[{"text": "Dose is 900 mg", "sources": ["S1"]},
                                           {"text": "Dose is 500 mg", "sources": ["S1"]}])
    out = ai.verify(raw, _ev("The dose is 500 mg."))
    assert [p["text"] for p in out["points"]] == ["Dose is 500 mg"] and out["dropped"] == 1


def test_a_chain_with_an_unsupported_number_is_dropped():
    out = ai.verify(_raw("**Yes.**", ["S1"], chain=["Blocks 90% of receptors", "Lowers pressure"],
                         chain_sources=["S1"]), _ev("It blocks receptors."))
    assert out["chain"] == [] and out["dropped"] == 1


def test_written_citations_are_stripped():
    out = ai.verify(_raw("**Yes** [S1, S2].", ["S1"]), _ev("It works."))
    assert out["headline"] == "**Yes**."


def test_lengths_are_capped_whatever_the_model_sends():
    raw = _raw("**Yes.**", ["S1"], points=[{"text": f"Point {w}", "sources": ["S1"]} for w in "abcde"],
               safety=[{"level": "amber", "text": f"Care {w}", "sources": ["S1"]} for w in "abcd"],
               followups=[f"Next {w}?" for w in "abcdef"])
    out = ai.verify(raw, _ev("It works."))
    assert (len(out["points"]), len(out["safety"]), len(out["followups"])) == \
           (ai.MAX_POINTS, ai.MAX_SAFETY, ai.MAX_FOLLOWUPS)


def test_a_row_only_one_drug_has_data_for_is_dropped():
    raw = _raw("**They differ.**", ["S1"], table_columns=["Feature", "A", "B"],
               table_rows=[{"cells": ["Half-life", "Not stated in sources", "26 hours"], "sources": ["S1"]},
                           {"cells": ["Class", "SSRI", "SSRI"], "sources": ["S1"]}])
    out = ai.verify(raw, _ev("Half-life 26 hours."))
    assert [r["cells"][0] for r in out["table_rows"]] == ["Class"]


def test_a_row_with_the_wrong_number_of_cells_is_dropped():
    raw = _raw("**They differ.**", ["S1"], table_columns=["Feature", "A", "B"],
               table_rows=[{"cells": ["Class", "SSRI"], "sources": ["S1"]}])
    out = ai.verify(raw, _ev("It works."))
    assert out["table_rows"] == [] and out["table_columns"] == []


@pytest.mark.parametrize("kind", ["clarify", "off_topic", "crisis", "refuse"])
def test_other_kinds_carry_nothing_but_their_line(kind):
    out = ai.verify(_raw("A line.", kind=kind, points=[{"text": "x", "sources": []}]), _ev())
    assert out["kind"] == kind and out["points"] == [] and out["headline"] == "A line."


def test_general_means_the_answer_has_no_source():
    assert ai.verify(_raw("**Agonists activate receptors.**"), _ev())["general"] is True
    assert ai.verify(_raw("**Yes.**", ["S1"]), _ev("It works."))["general"] is False


# --------------------------------------------------------------- Evidence
def test_evidence_gives_one_id_per_text_and_skips_blanks():
    ev = ai.Evidence()
    shown = ev.format([{"text": "Dose  is\n500 mg.", "src": "FDA label", "section": "Dosage"},
                       {"text": "dose is 500 mg.", "src": "Wikipedia"}, {"text": "  ", "src": "X"}])
    assert list(ev.items) == ["S1"] and shown.count("[S1]") == 2
    assert "(FDA label · Dosage) Dose is 500 mg." in shown


# ---------------------------------------------------------- the conversation
def _text(raw, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="text", text=json.dumps(raw))],
              usage=NS(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0))


def _tool_use(name, inp, id_="t1"):
    return NS(stop_reason="tool_use", content=[NS(type="tool_use", name=name, input=inp, id=id_)],
              usage=NS(input_tokens=500, output_tokens=50, cache_read_input_tokens=0))


class FakeClient:
    """Plays back scripted responses and records each request."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(dict(kw, messages=list(kw["messages"])))
        return self.responses.pop(0)


def _ask(client, question="q", **kw):
    kw.setdefault("tools", {})
    return ai.answer(question, client=client, today="2026-10-30", **kw)


def test_one_call_when_the_evidence_is_already_there():
    client = FakeClient(_text(_raw("**500 mg.**", ["S1"])))
    res = _ask(client, prefetched=[{"text": "The dose is 500 mg.", "src": "FDA label"}])
    assert res["status"] == "ok" and res["usage"]["calls"] == 1
    assert "[S1]" in client.requests[0]["messages"][-1]["content"]


def test_request_uses_the_settings_this_model_needs():
    client = FakeClient(_text(_raw()))
    _ask(client)
    req = client.requests[0]
    assert req["model"] == ai.MODEL and req["output_config"]["effort"] == ai.EFFORT
    assert req["output_config"]["format"]["schema"] is ai.ANSWER_SCHEMA
    for not_accepted in ("temperature", "top_p", "top_k", "fallbacks"):
        assert not_accepted not in req


def test_tool_results_reach_the_model_and_can_be_cited():
    client = FakeClient(_tool_use("get_drug_evidence", {"drug": "x", "topics": [], "keywords": []}),
                        _text(_raw("**500 mg.**", ["S1"])))
    res = _ask(client, tools={"get_drug_evidence": lambda inp: [{"text": "Dose 500 mg.", "src": "FDA label"}]})
    assert res["status"] == "ok" and res["answer"]["headline_sources"] == ["S1"]
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and "[S1]" in result["content"]


def test_a_failing_source_is_reported_to_the_model_not_raised():
    def broken(inp):
        raise ConnectionError("down")
    client = FakeClient(_tool_use("get_drug_evidence", {"drug": "x"}), _text(_raw()))
    assert _ask(client, tools={"get_drug_evidence": broken})["status"] == "ok"
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_nothing_found_is_said_plainly():
    client = FakeClient(_tool_use("search_encyclopedia", {"subject": "x"}), _text(_raw()))
    _ask(client, tools={"search_encyclopedia": lambda inp: []})
    assert "Nothing found" in client.requests[1]["messages"][-1]["content"][0]["content"]


def test_the_last_round_cannot_ask_for_more_tools():
    looping = [_tool_use("search_encyclopedia", {"subject": "x"}, f"t{i}") for i in range(ai.MAX_ROUNDS - 1)]
    client = FakeClient(*looping, _text(_raw()))
    assert _ask(client, tools={"search_encyclopedia": lambda inp: []})["status"] == "ok"
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
    assert "tool_choice" not in client.requests[0]


def test_a_refusal_is_not_shown_as_an_answer():
    res = _ask(FakeClient(NS(stop_reason="refusal", content=[], usage=None)))
    assert res["status"] == "refusal" and res["answer"] is None


def test_a_cut_off_answer_is_not_shown():
    res = _ask(FakeClient(_text(_raw(), stop="max_tokens")))
    assert res["status"] == "failed" and res["answer"] is None


def test_unparseable_output_is_not_shown():
    bad = NS(stop_reason="end_turn", content=[NS(type="text", text="{not json")], usage=None)
    assert _ask(FakeClient(bad))["status"] == "failed"


def test_thinking_blocks_are_skipped_when_reading_the_answer():
    resp = _text(_raw())
    resp.content.insert(0, NS(type="thinking", thinking="…"))
    assert _ask(FakeClient(resp))["status"] == "ok"


def test_a_wrong_number_gets_one_chance_to_be_corrected():
    client = FakeClient(_text(_raw("**900 mg.**", ["S1"])), _text(_raw("**500 mg.**", ["S1"])))
    res = _ask(client, prefetched=[{"text": "The dose is 500 mg.", "src": "FDA label"}])
    assert res["status"] == "ok" and res["answer"]["headline"] == "**500 mg.**"
    assert "could not verify" in client.requests[1]["messages"][-1]["content"]


def test_a_number_still_wrong_after_correction_is_never_shown():
    client = FakeClient(_text(_raw("**900 mg.**", ["S1"])), _text(_raw("**800 mg.**", ["S1"])))
    res = _ask(client, prefetched=[{"text": "The dose is 500 mg.", "src": "FDA label"}])
    assert res["status"] == "unverified" and res["answer"] is None and res["usage"]["calls"] == 2


def test_history_comes_before_the_question():
    client = FakeClient(_text(_raw()))
    _ask(client, "and in pregnancy?", history=[("half life of diazepam", "**30-56 hours**")])
    roles = [(m["role"], m["content"]) for m in client.requests[0]["messages"]]
    assert roles == [("user", "half life of diazepam"), ("assistant", "**30-56 hours**"),
                     ("user", "and in pregnancy?")]


def test_cost_is_estimated_at_list_price():
    assert ai.cost_usd({"input_tokens": 1_000_000, "output_tokens": 1_000_000}) == pytest.approx(0.60)
    assert ai.cost_sgd({"input_tokens": 1_000_000, "output_tokens": 1_000_000}) == pytest.approx(0.60 * ai.SGD_PER_USD)


def test_tokens_already_spent_survive_a_later_api_error():
    spent = {}
    client = FakeClient(_tool_use("get_drug_evidence", {"drug": "x"}))      # the next call raises
    with pytest.raises(IndexError):
        _ask(client, tools={"get_drug_evidence": lambda i: []}, usage=spent)
    assert spent["input_tokens"] == 500 and spent["output_tokens"] == 50


def test_prompt_carries_the_market_and_reader():
    sg = ai.system_prompt(region="SG", audience="Everyday", today="2026-10-30")
    assert "Singapore" in sg and "995" in sg and "no medical training" in sg and "2026-10-30" in sg
    assert "911" in ai.system_prompt(region="US", audience="Clinician", today="2026-10-30")


def test_unknown_audience_uses_the_default():
    assert ai.AUDIENCES[ai.DEFAULT_AUDIENCE] in ai.system_prompt(region="SG", audience="?", today="x")


# ---------------------------------------------------------------- spend limits
@pytest.fixture
def fresh_limits():
    ai._QUOTA.update(day="", count=0)
    ai._ANSWERS.clear()
    yield
    ai._QUOTA.update(day="", count=0)
    ai._ANSWERS.clear()


def test_daily_limit_stops_and_resets_next_day(fresh_limits):
    assert [ai.take_quota("d1", 2) for _ in range(3)] == [True, True, False]
    assert ai.take_quota("d2", 2) is True


def test_answer_cache_expires_and_evicts(fresh_limits, monkeypatch):
    ai.remember_answer("k", {"v": 1})
    assert ai.cached_answer("k") == {"v": 1}
    monkeypatch.setattr(ai.time, "time", lambda: 10 ** 12)
    assert ai.cached_answer("k") is None
    monkeypatch.undo()
    for i in range(ai.ANSWER_CACHE_SIZE + 5):
        ai.remember_answer(i, i)
    assert ai.cached_answer(0) is None and ai.cached_answer(ai.ANSWER_CACHE_SIZE + 4) == ai.ANSWER_CACHE_SIZE + 4


# ------------------------------------------------------------------- in the app
def _ask_app(*questions):
    at = AppTest.from_file(APP, default_timeout=120).run()
    for q in questions:
        at.chat_input[0].set_value(q).run()
    assert not at.exception
    return at


def _all_text(at):
    msg = at.chat_message[-1]
    return " ".join(str(getattr(e, "value", "")) for e in msg)


@pytest.fixture
def api(monkeypatch, fresh_limits):
    """The app with a key set, and ai.answer scripted: no call leaves this machine."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(ai, "make_client", lambda key: object())
    state = NS(calls=[], result=None, spent={})

    def fake_answer(question, **kw):
        state.calls.append((question, kw))
        kw["usage"].update(state.spent)
        if isinstance(state.result, Exception):
            raise state.result
        return state.result
    monkeypatch.setattr(ai, "answer", fake_answer)
    return state


def _ok(**fields):
    ev = {"S1": {"text": "It works by blocking receptors.", "src": "FDA label", "url": "https://e.org", "section": ""}}
    answer = ai.verify(_raw("**Blocks receptors.**", ["S1"], **fields), _ev("It works by blocking receptors."))
    return {"status": "ok", "answer": answer, "evidence": ev, "usage": {}, "tools": [], "raw": None}


def test_no_key_means_no_ai_and_no_mention_of_it(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _ask_app()
    assert "ai_audience" not in [r.key for r in at.sidebar.radio]


def test_the_card_shows_the_answer_with_its_source_and_tag(api):
    api.result = _ok(safety=[{"level": "red", "text": "Stop if a rash occurs.", "sources": ["S1"]}],
                     followups=["Side effects of x?"])
    at = _ask_app(NO_LOOKUP_Q)
    body = " ".join(e.value for e in at.chat_message[-1].markdown)
    assert "Blocks receptors" in body and "ai-red" in body and "FDA label" in body
    assert "Side effects of x?" in [b.label for b in at.chat_message[-1].button]
    assert any("AI summary of the linked sources" in c.value for c in at.chat_message[-1].caption)
    assert "ai_audience" in [r.key for r in at.sidebar.radio]


def test_an_answer_without_a_source_says_so(api):
    answer = ai.verify(_raw("**Agonists activate receptors.**"), _ev())
    api.result = {"status": "ok", "answer": answer, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    at = _ask_app(NO_LOOKUP_Q)
    assert any("AI answer, not from a source" in c.value for c in at.chat_message[-1].caption)


def test_off_topic_has_no_source_caption(api):
    answer = ai.verify(_raw("This app only covers medicines.", kind="off_topic"), _ev())
    api.result = {"status": "ok", "answer": answer, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    at = _ask_app(NO_LOOKUP_Q)
    assert not any("AI" in c.value for c in at.chat_message[-1].caption)


@pytest.mark.parametrize("status", ["unverified", "failed", "refusal"])
def test_an_answer_that_cant_be_shown_falls_back_to_the_sources(api, status):
    api.result = {"status": status, "answer": None, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    at = _ask_app(NO_LOOKUP_Q)
    assert len(api.calls) == 1
    assert any("source text directly" in c.value for c in at.chat_message[-1].caption)


def test_an_api_failure_falls_back_to_the_sources(api):
    api.result = ConnectionError("no route")
    at = _ask_app(NO_LOOKUP_Q)
    assert any("isn't available" in c.value for c in at.chat_message[-1].caption)


@pytest.mark.parametrize("q", ["i want to kill myself", "how do i make meth at home", "hello", "help"])
def test_safety_wording_and_small_talk_never_reach_the_model(api, q):
    _ask_app(q)
    assert api.calls == []


def test_a_repeated_question_costs_nothing(api):
    api.result = _ok()
    _ask_app(NO_LOOKUP_Q)
    at = AppTest.from_file(APP, default_timeout=120).run()      # another visitor, same question
    at.chat_input[0].set_value(NO_LOOKUP_Q).run()
    assert len(api.calls) == 1
    assert "Blocks receptors" in " ".join(e.value for e in at.chat_message[-1].markdown)


def _captions(at):
    return " ".join(c.value for c in at.chat_message[-1].caption)


def test_an_answer_shows_its_tokens_and_cost_in_sgd(api):
    api.result, api.spent = _ok(), {"input_tokens": 3000, "output_tokens": 200}
    assert "3,200 tokens · ≈ S\\$0.0005" in _captions(_ask_app(NO_LOOKUP_Q))


def test_off_topic_shows_what_it_spent(api):
    answer = ai.verify(_raw("This app only covers medicines.", kind="off_topic"), _ev())
    api.result = {"status": "ok", "answer": answer, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    api.spent = {"input_tokens": 900, "output_tokens": 40}
    assert "940 tokens" in _captions(_ask_app(NO_LOOKUP_Q))


def test_a_failed_attempt_still_shows_what_it_spent(api):
    api.result = {"status": "unverified", "answer": None, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    api.spent = {"input_tokens": 2000, "output_tokens": 100}
    assert "2,100 tokens" in _captions(_ask_app(NO_LOOKUP_Q))


def test_an_api_error_still_shows_what_it_spent(api):
    api.result, api.spent = ConnectionError("no route"), {"input_tokens": 500, "output_tokens": 50}
    assert "550 tokens" in _captions(_ask_app(NO_LOOKUP_Q))


def test_a_saved_answer_shows_no_tokens(api):
    api.result, api.spent = _ok(), {"input_tokens": 3000, "output_tokens": 200}
    _ask_app(NO_LOOKUP_Q)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value(NO_LOOKUP_Q).run()
    assert "0 tokens · S\\$0 (saved answer)" in _captions(at)


def test_a_fixed_reply_shows_no_tokens(api):
    assert "0 tokens · S\\$0" in _captions(_ask_app("how do i make meth at home"))
    assert api.calls == []


def test_no_key_shows_no_token_count(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "tokens" not in _captions(_ask_app("how do i make meth at home"))


def test_daily_limit_falls_back_without_calling(api, monkeypatch):
    monkeypatch.setenv("AI_DAILY_LIMIT", "0")
    api.result = _ok()
    at = _ask_app(NO_LOOKUP_Q)
    assert api.calls == []
    assert any("limit has been reached" in c.value for c in at.chat_message[-1].caption)


def test_session_limit_falls_back_without_calling(api, monkeypatch):
    monkeypatch.setenv("AI_SESSION_LIMIT", "1")
    api.result = _ok()
    at = _ask_app(NO_LOOKUP_Q, "what's the capital of spain")
    assert len(api.calls) == 1
    assert any("limit has been reached" in c.value for c in at.chat_message[-1].caption)


def test_a_follow_up_sends_the_earlier_exchange(api):
    api.result = _ok()
    _ask_app(NO_LOOKUP_Q, "what's the capital of spain")
    assert api.calls[1][1]["history"] == [(NO_LOOKUP_Q, "**Blocks receptors.**")]


def test_the_key_is_never_in_what_the_page_shows(api):
    api.result = ConnectionError("no route")
    at = _ask_app(NO_LOOKUP_Q)
    assert "test-key-not-real" not in " ".join(str(getattr(e, "value", "")) for e in at.main)
