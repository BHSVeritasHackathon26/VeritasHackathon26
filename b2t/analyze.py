"""Turns the saved results into tables and graphs for the write-up. Example:

    python -m b2t.analyze

It only reads files in results/ (and, if present, the dataset authors' table of
sentence types). It trains nothing and changes no results. Graphs are saved as PNG
images in figures/, and every number behind them is saved in results/analysis.json.

Sections that need a file which does not exist yet are skipped, so this can be
re-run as more results arrive.

Sentence types come from `t15_copyTaskData_description.csv` in the dataset authors'
GitHub repo (github.com/Neuroprosthetics-Lab/nejm-brain-to-text, in data/). Save it
to external/authors_repo/ to enable that section.
"""

import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from b2t.config import REPO_ROOT
from b2t.metrics import edit_distance, normalize_text

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
TYPE_TABLE = REPO_ROOT / "external" / "authors_repo" / "t15_copyTaskData_description.csv"

# Colours: one fixed colour per role, used the same way in every graph.
OURS = "#2a78d6"       # our model / the main series
SECOND = "#eb6834"     # the second series, or "not seen in training"
THIRD = "#1baf7a"
MUTED = "#a3a29b"      # baselines and reference lines
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUES = ["#9cc3ee", "#5f9de0", "#2a78d6", "#134a8e"]  # light to dark = less to more


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def new_chart(title, subtitle=None, size=(7.5, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=9, color=INK_SOFT)
    return fig, ax


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGURES / name, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved figures/{name}")


def legend_below(ax, handles=None, drop=0.2):
    """A legend in one row under the graph, where it cannot cover the data."""
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_SOFT,
              loc="upper center", bbox_to_anchor=(0.5, -drop), ncols=4)


def percent_axis(ax, label):
    ax.set_ylabel(label, color=INK_SOFT, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(bottom=0)


def add_edits(rows):
    """Work out, for every sentence, how many phonemes and words were wrong."""
    for r in rows:
        truth = normalize_text(r["truth"]).split()
        r["n_words"] = len(truth)
        r["n_phonemes"] = len(r["true_phonemes"])
        r["phoneme_edits"] = edit_distance(r["predicted_phonemes"], r["true_phonemes"])
        r["dict_edits"] = edit_distance(r["dictionary"].split(), truth)
        r["llm_edits"] = (edit_distance(normalize_text(r["llm"]).split(), truth)
                          if r.get("llm") is not None else None)
    return rows


def rates(rows):
    """Aggregate error rates for a group of sentences (total edits / total length)."""
    words = sum(r["n_words"] for r in rows)
    has_llm = all(r["llm_edits"] is not None for r in rows)
    return {
        "sentences": len(rows),
        "phoneme_error": sum(r["phoneme_edits"] for r in rows) / sum(r["n_phonemes"] for r in rows),
        "word_error_dictionary": sum(r["dict_edits"] for r in rows) / words,
        "word_error_llm": sum(r["llm_edits"] for r in rows) / words if has_llm else None,
    }


def grouped(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {name: rates(group) for name, group in groups.items()}


def show_table(title, table, order=None):
    print(f"\n{title}")
    print(f"  {'group':28s} {'sentences':>9s} {'phonemes':>9s} {'words (dict)':>13s} {'words (LLM)':>12s}")
    for name in order or table:
        if name not in table:
            continue
        t = table[name]
        llm = f"{t['word_error_llm']:.1%}" if t["word_error_llm"] is not None else "-"
        print(f"  {str(name):28s} {t['sentences']:9d} {t['phoneme_error']:9.1%} "
              f"{t['word_error_dictionary']:13.1%} {llm:>12s}")


def sentence_types():
    """(session, block) -> sentence type, from the authors' table. None if missing."""
    if not TYPE_TABLE.exists():
        return None
    with open(TYPE_TABLE) as f:
        return {(row["Date"].replace("-", "."), int(row["Block number"])): row["Corpus"]
                for row in csv.DictReader(f)}


def session_day(session):
    return session.split(".", 1)[1]  # "t15.2023.08.13" -> "2023.08.13"


def main():
    out = {}
    val = load("run_a_val.json")
    tune = load("run_a_tune.json")
    base_val = load("linear_val.json")
    run_a = load("run_a_run.json")
    audit = load("audit_t15.json")
    if not val:
        raise SystemExit("results/run_a_val.json not found; nothing to analyse.")
    rows = add_edits(val["sentences"])
    out["run_a_val_overall"] = rates(rows)
    print(f"Run A on val: {len(rows)} sentences")

    # ---- Baseline against our model, on the same held-out sentences ----
    if base_val:
        base_rows = add_edits(base_val["sentences"])
        out["linear_val_overall"] = rates(base_rows)
        show_table("Baseline against our model (val)",
                   {"linear baseline": out["linear_val_overall"], "Run A (GRU)": out["run_a_val_overall"]})
        fig, ax = new_chart("Our model against the simple baseline",
                            "Error on 1,426 held-out sentences. Lower is better.")
        labels = ["Phonemes", "Words\n(dictionary decoder)", "Words\n(language-model decoder)"]
        keys = ["phoneme_error", "word_error_dictionary", "word_error_llm"]
        for offset, (name, colour, data) in enumerate((("Linear baseline", MUTED, out["linear_val_overall"]),
                                                       ("Our model (GRU)", OURS, out["run_a_val_overall"]))):
            # A result file scored without the language model has no value for it.
            shown = [(i + (offset - 0.5) * 0.36, 100 * data[k]) for i, k in enumerate(keys)
                     if data[k] is not None]
            ax.bar([x for x, _ in shown], [y for _, y in shown], width=0.32, color=colour, label=name)
            for x, y in shown:
                ax.text(x, y + 1.5, f"{y:.1f}%", ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(3), labels)
        percent_axis(ax, "error rate")
        ax.set_ylim(0, 110)
        legend_below(ax)
        save(fig, "baseline_vs_model.png")

    # ---- Sentence types ----
    types = sentence_types()
    seen_in_training = {"Switchboard", "Freq words", "50-Word"}
    if types:
        for r in rows:
            r["type"] = types.get((session_day(r["session"]), r["block"]), "unknown")
        by_type = grouped(rows, lambda r: r["type"])
        out["val_by_sentence_type"] = by_type
        order = sorted(by_type, key=lambda t: by_type[t]["word_error_llm"])
        show_table("Val by sentence type", by_type, order)
        seen = [r for r in rows if r["type"] in seen_in_training]
        unseen = [r for r in rows if r["type"] not in seen_in_training]
        out["val_types_seen_in_training"] = rates(seen)
        out["val_types_not_seen_in_training"] = rates(unseen) if unseen else None
        show_table("Val, by whether the sentence TYPE appeared in training",
                   {"type seen in training": rates(seen), "type never in training": rates(unseen)})

        fig, ax = new_chart("Error by kind of sentence",
                            "Word error (language-model decoder) on held-out sentences.")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        names = {"Switchboard": "Everyday conversation", "Freq words": "Frequent-word sentences",
                 "Openwebtext": "Written web text", "Harvard": "Harvard sentences",
                 "Random": "Random word lists", "50-Word": "50-word vocabulary"}
        for i, t in enumerate(order):
            value = 100 * by_type[t]["word_error_llm"]
            ax.barh(i, value, height=0.55, color=OURS if t in seen_in_training else SECOND)
            ax.text(value + 1, i, f"{value:.1f}%  ({by_type[t]['sentences']} sentences)",
                    va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(order)), [names.get(t, t) for t in order])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xlim(0, max(100 * by_type[t]["word_error_llm"] for t in order) * 1.45)
        legend_below(ax, handles=[Patch(color=OURS, label="Kind of sentence seen in training"),
                                  Patch(color=SECOND, label="Kind never seen in training")])
        save(fig, "error_by_sentence_type.png")
    else:
        print("\n(sentence-type table not found; skipping that section)")

    # ---- Sentence length ----
    bins = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 99)]
    label = lambda lo, hi: f"{lo}-{hi}" if hi < 99 else f"{lo}+"
    def length_bin(r):
        return next(label(lo, hi) for lo, hi in bins if lo <= r["n_words"] <= hi)
    by_length = grouped(rows, length_bin)
    out["val_by_sentence_length"] = by_length
    length_order = [label(lo, hi) for lo, hi in bins if label(lo, hi) in by_length]
    show_table("Val by sentence length (words in the true sentence)", by_length, length_order)
    fig, ax = new_chart("Error by sentence length",
                        "Word error on held-out sentences, by words in the sentence "
                        "(number of sentences in brackets).")
    for offset, (name, colour, key) in enumerate((("Dictionary decoder", MUTED, "word_error_dictionary"),
                                                  ("Language-model decoder", OURS, "word_error_llm"))):
        xs = [i + (offset - 0.5) * 0.36 for i in range(len(length_order))]
        ys = [100 * by_length[b][key] for b in length_order]
        ax.bar(xs, ys, width=0.32, color=colour, label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.6, f"{y:.0f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(length_order)),
                  [f"{b} words\n({by_length[b]['sentences']})" for b in length_order])
    percent_axis(ax, "word error rate")
    legend_below(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    save(fig, "error_by_sentence_length.png")

    # ---- Wording seen in training or not ----
    if audit:
        repeated = {(t["session"], t["block"], t["trial"])
                    for t in audit["val_sentences_with_wording_seen_in_training"]["trials"]}
        by_repeat = grouped(rows, lambda r: "wording also in training"
                            if (r["session"], r["block"], r["trial"]) in repeated else "new wording")
        out["val_by_repeated_wording"] = by_repeat
        show_table("Val by whether the exact wording appeared in training", by_repeat)

    # ---- What the language model changes ----
    better = [r for r in rows if r["llm_edits"] < r["dict_edits"]]
    worse = [r for r in rows if r["llm_edits"] > r["dict_edits"]]
    same = [r for r in rows if r["llm_edits"] == r["dict_edits"]]
    out["language_model_effect"] = {
        "sentences_improved": len(better), "sentences_worse": len(worse), "sentences_unchanged": len(same),
        "word_errors_dictionary": sum(r["dict_edits"] for r in rows),
        "word_errors_llm": sum(r["llm_edits"] for r in rows),
        "sentences_perfect_dictionary": sum(r["dict_edits"] == 0 for r in rows),
        "sentences_perfect_llm": sum(r["llm_edits"] == 0 for r in rows),
        "sentences_perfect_phonemes": sum(r["phoneme_edits"] == 0 for r in rows),
    }
    e = out["language_model_effect"]
    print(f"\nLanguage-model decoder against dictionary decoder (val):")
    print(f"  improved {len(better)} sentences, made {len(worse)} worse, left {len(same)} the same")
    print(f"  total word errors: {e['word_errors_dictionary']} -> {e['word_errors_llm']}")
    print(f"  sentences with every word right: {e['sentences_perfect_dictionary']} -> "
          f"{e['sentences_perfect_llm']} of {len(rows)}")
    print(f"  sentences with every phoneme right: {e['sentences_perfect_phonemes']}")

    # ---- Tune against val ----
    if tune:
        out["run_a_tune_overall"] = rates(add_edits(tune["sentences"]))
        show_table("Run A: tune against val",
                   {"tune": out["run_a_tune_overall"], "val": out["run_a_val_overall"]})

    # ---- By recording session (useful beside the drift experiment) ----
    by_session = grouped(rows, lambda r: r["session"])
    out["val_by_session"] = by_session

    # ---- Training curve ----
    if run_a:
        fig, ax = new_chart("How the model improved during training",
                            "Run A: phoneme error on the tuning sentences, checked every 4,000 batches.")
        steps = [h["step"] / 1000 for h in run_a["history"]]
        ys = [100 * h["tune_per"] for h in run_a["history"]]
        ax.plot(steps, ys, color=OURS, linewidth=2)
        best = min(run_a["history"], key=lambda h: h["tune_per"])
        ax.plot([best["step"] / 1000], [100 * best["tune_per"]], "o", color=OURS, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"best: {100 * best['tune_per']:.1f}%", (best["step"] / 1000, 100 * best["tune_per"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
        ax.annotate(f"{ys[0]:.1f}%", (steps[0], ys[0]), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=9, color=INK)
        ax.set_xlabel("training batches (thousands)", color=INK_SOFT, fontsize=9)
        percent_axis(ax, "phoneme error rate")
        save(fig, "training_curve.png")

    # ---- Data-amount experiment ----
    fractions = []
    for path in sorted(RESULTS.glob("frac_*_run.json")):
        r = json.loads(path.read_text())
        best = min(r["history"], key=lambda h: h["tune_per"])
        tune_file = load(path.name.replace("_run.json", "_tune.json"))
        val_file = load(path.name.replace("_run.json", "_val.json"))
        fractions.append({"fraction": r["train_fraction"], "sentences": r["n_train_trials"],
                          "batches": r["args"]["batches"], "best_tune_phoneme_error": r["best_tune_per"],
                          "best_at_batch": best["step"], "history": r["history"],
                          "tune_word_error_dictionary": tune_file["word_error_rate_dictionary"] if tune_file else None,
                          "val_phoneme_error": val_file["phoneme_error_rate"] if val_file else None,
                          "val_word_error_dictionary": val_file["word_error_rate_dictionary"] if val_file else None})
    if fractions:
        fractions.sort(key=lambda f: f["sentences"])
        out["data_amount"] = [{k: v for k, v in f.items() if k != "history"} for f in fractions]
        print("\nData-amount experiment (tune, phoneme error)")
        for f in fractions:
            words = (f", word error (dictionary) {f['tune_word_error_dictionary']:.1%}"
                     if f["tune_word_error_dictionary"] is not None else "")
            print(f"  {f['fraction']:4.0%}  {f['sentences']:5d} sentences: {f['best_tune_phoneme_error']:.1%} "
                  f"(best at batch {f['best_at_batch']}){words}")
        if all(f["val_phoneme_error"] is not None for f in fractions):
            print("Data-amount experiment (val, dictionary decoder)")
            for f in fractions:
                print(f"  {f['fraction']:4.0%}  {f['sentences']:5d} sentences: phonemes {f['val_phoneme_error']:.1%}, "
                      f"words {f['val_word_error_dictionary']:.1%}")

        xs = [f["sentences"] for f in fractions]
        for split, phoneme_key, word_key, where, file_name in (
                ("tune", "best_tune_phoneme_error", "tune_word_error_dictionary", "tuning sentences", "data_amount.png"),
                ("val", "val_phoneme_error", "val_word_error_dictionary", "held-out sentences", "data_amount_val.png")):
            if any(f[phoneme_key] is None for f in fractions):
                continue
            fig, ax = new_chart("More training sentences, lower error",
                                f"Same model trained on a share of the training sentences, 40,000 batches each. "
                                f"Scored on the {where}.")
            ys = [100 * f[phoneme_key] for f in fractions]
            have_words = all(f[word_key] is not None for f in fractions)
            top = max(ys)
            if have_words:
                ws = [100 * f[word_key] for f in fractions]
                top = max(ws)
                ax.plot(xs, ws, color=SECOND, linewidth=2, marker="o", markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label="Word error (dictionary decoder)")
                for x, w in zip(xs, ws):
                    ax.annotate(f"{w:.1f}%", (x, w), textcoords="offset points", xytext=(8, 8),
                                fontsize=9, color=INK)
            ax.plot(xs, ys, color=OURS, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label="Phoneme error")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(8, 8),
                            fontsize=9, color=INK)
            ax.set_xticks(xs, [f"{x:,}\n({f['fraction']:.0%})" for x, f in zip(xs, fractions)])
            ax.set_xlabel("training sentences (share of all training data)", color=INK_SOFT, fontsize=9)
            ax.set_xlim(0, max(xs) * 1.1)
            percent_axis(ax, "error rate")
            ax.set_ylim(0, top * 1.2)
            if have_words:
                legend_below(ax, drop=0.3)
            save(fig, file_name)

        fig, ax = new_chart("Less data is used up sooner",
                            "Phoneme error during training, for four amounts of training data.")
        for f, colour in zip(fractions, BLUES[-len(fractions):] if len(fractions) <= 4 else BLUES * 2):
            steps = [h["step"] / 1000 for h in f["history"]]
            ys = [100 * h["tune_per"] for h in f["history"]]
            ax.plot(steps, ys, color=colour, linewidth=2, label=f"{f['fraction']:.0%} of the data")
            ax.annotate(f"{f['sentences']:,} sentences: best {100 * f['best_tune_phoneme_error']:.1f}%",
                        (steps[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=9, color=INK)
        ax.set_xlabel("training batches (thousands)", color=INK_SOFT, fontsize=9)
        ax.set_xlim(0, max(h["step"] for h in fractions[0]["history"]) / 1000 * 1.5)
        ax.set_xticks(range(0, 41, 10))
        percent_axis(ax, "phoneme error rate")
        legend_below(ax)
        save(fig, "training_by_data_amount.png")

    # ---- Training length: same model and data, different numbers of batches ----
    lengths = {}
    for path in sorted(RESULTS.glob("*_run.json")):
        r = json.loads(path.read_text())
        full = (r.get("model") == "gru" and r.get("train_fraction", 1.0) == 1.0
                and len(r.get("trained_on_sessions", [])) == len(r.get("all_sessions", []))
                and r["args"].get("batch_size") == 32)
        if full:
            lengths[r["args"]["batches"]] = {"file": path.name, "best_tune_phoneme_error": r["best_tune_per"]}
    if len(lengths) >= 2:
        out["training_length"] = {str(k): v for k, v in sorted(lengths.items())}
        print("\nTraining length (all data, batch size 32)")
        for batches, v in sorted(lengths.items()):
            print(f"  {batches:7,d} batches: {v['best_tune_phoneme_error']:.1%}  ({v['file']})")
        fig, ax = new_chart("Training for longer",
                            "Same model and data, trained for different numbers of batches.", size=(6, 4.2))
        xs = sorted(lengths)
        ys = [100 * lengths[b]["best_tune_phoneme_error"] for b in xs]
        ax.bar(range(len(xs)), ys, width=0.5, color=OURS)
        for i, y in enumerate(ys):
            ax.text(i, y + 0.3, f"{y:.1f}%", ha="center", fontsize=10, color=INK)
        ax.set_xticks(range(len(xs)), [f"{b:,}\nbatches" for b in xs])
        percent_axis(ax, "phoneme error rate (tuning sentences)")
        ax.set_ylim(0, max(ys) * 1.2)
        save(fig, "training_length.png")

    # ---- Drift experiment ----
    for path in sorted(RESULTS.glob("drift_*.json")):
        d = json.loads(path.read_text())
        if "per_session" not in d:  # e.g. drift_run.json, the drift model's training record
            continue
        out["drift"] = {"summary": d["summary"], "last_training_session": d["last_training_session"]}
        print(f"\nDrift experiment ({path.name}), all later sessions together")
        for s in d["summary"]:
            print(f"  {s['n_sentences']:3d} recalibration sentences: phoneme error {s['phoneme_error_rate']:.1%}, "
                  f"word error (dictionary) {s['word_error_rate_dictionary']:.1%}  ({s['n_sessions']} sessions)")
        fig, ax = new_chart("Accuracy after training stops",
                            f"Model trained on sessions up to {session_day(d['last_training_session'])}, "
                            "tested on each later session's held-out sentences.", size=(8, 4.4))
        # Reference: our final model, which trained on these later sessions too.
        tested = {r["session"]: r["days_since_training"] for r in d["per_session"]}
        ref = sorted((days / 30.44, 100 * by_session[s]["phoneme_error"])
                     for s, days in tested.items() if s in by_session)
        if ref:
            ax.plot([p[0] for p in ref], [p[1] for p in ref], color=MUTED, linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label="Model trained on these sessions too (Run A)")
            ref_rows = [r for r in rows if r["session"] in tested]
            out["drift"]["run_a_on_same_sessions"] = rates(ref_rows)
        early = [r for r in d["per_session"] if r["days_since_training"] <= 90]
        late = [r for r in d["per_session"] if r["days_since_training"] > 365]
        def pooled(group, n):
            g = [r for r in group if r["n_sentences"] == n]
            return sum(r["phoneme_error_rate"] * r["n_val_trials"] for r in g) / max(1, sum(r["n_val_trials"] for r in g))
        out["drift"]["approx_by_period"] = {
            "within_3_months": {str(n): pooled(early, n) for n in sorted({r["n_sentences"] for r in early})},
            "after_12_months": {str(n): pooled(late, n) for n in sorted({r["n_sentences"] for r in late})}}
        sizes = sorted({r["n_sentences"] for r in d["per_session"]})
        for n, colour in zip(sizes, (SECOND, THIRD, OURS)):
            pts = sorted((r["days_since_training"] / 30.44, 100 * r["phoneme_error_rate"])
                         for r in d["per_session"] if r["n_sentences"] == n)
            name = "No recalibration" if n == 0 else f"Recalibrated on {n} sentences"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, linewidth=2, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        ax.set_xlabel("months since the last training session", color=INK_SOFT, fontsize=9)
        percent_axis(ax, "phoneme error rate")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), ncols=2)
        ax.set_ylim(0, 90)
        save(fig, "drift.png")

    # ---- What to show the language model: three methods compared ----
    for path in sorted(RESULTS.glob("llm_compare_*.json")):
        c = json.loads(path.read_text())
        def method_error(key, ignore_apostrophes=False):
            clean = (lambda t: normalize_text(t).replace("'", "")) if ignore_apostrophes else normalize_text
            edits = sum(edit_distance(clean(r[key]).split(), clean(r["truth"]).split()) for r in c["sentences"])
            return edits / sum(len(clean(r["truth"]).split()) for r in c["sentences"])
        labels = {"dictionary": "Dictionary only\n(no language model)",
                  "candidates": "Dictionary candidates\n(the method we use)",
                  "phonemes": "Phonemes only",
                  "alternatives": "Phonemes with\nalternatives and odds"}
        methods = [m for m in labels if m in c["summary"]]
        table = {m: {"word_error": c["summary"][m]["word_error_rate"],
                     "word_error_ignoring_apostrophes": method_error(m, True),
                     "sentences_all_correct": c["summary"][m]["sentences_all_correct"],
                     "fell_back_to_dictionary": c["summary"][m]["fell_back_to_dictionary"],
                     "outputs_with_an_apostrophe": sum("'" in r[m] for r in c["sentences"])}
                 for m in methods}
        out[f"language_model_methods_{c['split']}"] = {"llm_model": c["llm_model"], "n_sentences": c["n_sentences"],
                                                       "methods": table}
        print(f"\nWhat the language model is shown ({c['split']}, {c['n_sentences']} sentences, {c['llm_model']})")
        for m in methods:
            t = table[m]
            print(f"  {m:13s} word error {t['word_error']:.1%}  (ignoring apostrophes {t['word_error_ignoring_apostrophes']:.1%})  "
                  f"{t['sentences_all_correct']} fully correct")
        if len(methods) < 3:
            continue  # a single-method run (e.g. the final val score) needs no graph
        where = "tuning" if c["split"] == "tune" else "held-out"
        fig, ax = new_chart("What should the language model be shown?",
                            f"Word error on {c['n_sentences']} {where} sentences. Same neural network and "
                            f"same language model each time.", size=(7.5, 4.0))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        order = sorted(methods, key=lambda m: -table[m]["word_error"])
        for i, m in enumerate(order):
            value = 100 * table[m]["word_error"]
            ax.barh(i, value, height=0.55, color=MUTED if m == "dictionary" else OURS)
            ax.text(value + 0.4, i, f"{value:.1f}%", va="center", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(order)), [labels[m] for m in order])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xlim(0, max(100 * table[m]["word_error"] for m in order) * 1.18)
        save(fig, f"language_model_methods_{c['split']}.png")

    # ---- Candidate examples for the write-up ----
    rng = random.Random(0)
    def example(r):
        return {"session": r["session"], "block": r["block"], "trial": r["trial"],
                "type": r.get("type"), "truth": normalize_text(r["truth"]),
                "phonemes": " ".join(r["predicted_phonemes"]),
                "dictionary": r["dictionary"], "llm": r["llm"],
                "word_errors_dictionary": r["dict_edits"], "word_errors_llm": r["llm_edits"]}
    medium = [r for r in rows if 4 <= r["n_words"] <= 10]
    picks = {
        "one_word_wrong": rng.sample([r for r in medium if r["llm_edits"] == 1 and r["dict_edits"] == 1], 4),
        "language_model_fixed_it": rng.sample([r for r in medium if r["llm_edits"] == 0 and r["dict_edits"] >= 1], 4),
        "language_model_made_it_worse": sorted((r for r in medium if r["llm_edits"] > r["dict_edits"]),
                                               key=lambda r: r["dict_edits"] - r["llm_edits"])[:4],
        "badly_wrong": sorted(medium, key=lambda r: -r["llm_edits"] / r["n_words"])[:4],
        "perfect": rng.sample([r for r in medium if r["llm_edits"] == 0 and r["dict_edits"] == 0 and r["n_words"] >= 7], 3),
    }
    if types:
        random_rows = [r for r in rows if r.get("type") == "Random"]
        picks["random_word_lists"] = rng.sample(random_rows, min(4, len(random_rows)))
    out["example_candidates"] = {k: [example(r) for r in v] for k, v in picks.items()}
    print("\nCandidate examples (val)")
    for group, items in out["example_candidates"].items():
        print(f"  [{group}]")
        for ex in items:
            print(f"    truth       {ex['truth']}")
            print(f"    dictionary  {ex['dictionary']}")
            print(f"    language    {ex['llm']}\n")

    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=1))
    print("Saved every number to results/analysis.json")


if __name__ == "__main__":
    main()
