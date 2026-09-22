"""Build display_index.html — the artifact that indexes the display items.

    python _build_index.py

Reads the PNGs beside it, embeds a WebP thumbnail of each as a data URI, and
writes a single self-contained page. The prose is here rather than in the HTML
so the page can be rebuilt after a figure is regenerated without re-typing it.
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "display_index.html"
THUMB_W = 760


def thumb(name):
    im = Image.open(HERE / name).convert("RGB")
    w, h = im.size
    im = im.resize((THUMB_W, max(1, round(h * THUMB_W / w))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=82, method=5)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- the items
# (file, order-chip, title, the number or claim it carries, notes[])
# `order-chip` is a real sequence only inside the first tier, which is a talk
# order; elsewhere it is None and the tier is just a named group.
TIERS = [
    dict(
        id="thesis",
        name="Carries the thesis",
        lead="""Four figures, in the order they make the argument. Every one of
        them is about <em>the same</em> single-drug, 19-locus task — the only
        task where the claim holds — and each names its own reference line, so
        none of them needs the others to be read fairly.""",
        items=[
            dict(
                file="results_10_11-two-more-measurements-the-gate-and-the-price.png",
                chip="1",
                title="Parameters vs. accuracy, both families",
                carries="<b>locusfusion 0.9242 at 0.659M params</b> against mdcnn 0.9246 at 6.655M and cisfusion 0.9231 at 59.29M — four orders of magnitude on one axis.",
                notes=[
                    "The strongest single slide for the theme: same accuracy, a tenth of mdcnn and ~1/90 of cisfusion.",
                    "It also carries Part III on the same axis — <code>additive</code>, a bare sum of per-variant weights at 0.050M, beats every <code>late_fusion</code> cell.",
                ],
                src="notebooks/results.ipynb §11",
            ),
            dict(
                file="results_12_12-where-it-lands.png",
                chip="2",
                title="locusfusion by modality set and task, paired against mdcnn",
                carries="Paired cell by cell at 19 loci: <b>+0.0022 mean</b>, ahead on four of five modality sets (−0.0009 on <code>all_modalities</code>).",
                notes=[
                    "The right panel is the honest head-to-head. The left panel shows the spread across modality sets is 0.0012 — the modalities have been absorbed into the token.",
                    "That flatness is why the §1 'best cell' tiles put mdcnn ahead: a max over five draws rewards the model that varies more. Say this before someone else finds it.",
                ],
                src="notebooks/results.ipynb §12",
            ),
            dict(
                file="results_11_part-iv-locusfusion-the-measurements-assembled.png",
                chip="3",
                title="The LocusFusionNet architecture",
                carries="Two-stage fusion: a locus's modalities fuse into one locus representation, then the 19 loci talk to each other. A susceptible isolate is 19 <code>[WT]</code> sentinels and nothing else.",
                notes=[
                    "Every shape and parameter count is read off a real <code>LocusFusionNet</code> built at the 19-locus configuration, so the diagram cannot drift from the model.",
                    "Regenerate with <code>python scripts/locusfusion_diagram.py</code>; <code>--check</code> asserts the drawn total against a run JSON. SVG is beside the PNG for scaling.",
                ],
                src="scripts/locusfusion_diagram.py",
            ),
            dict(
                file="results_01_1-where-the-project-stands.png",
                chip="4",
                title="Headline tiles",
                carries="0.9246 best established / 0.9242 best locusfusion / both above the published MD-CNN 0.9222 / 92 complete cells over 993 drug-runs.",
                notes=[
                    "Use it to open or to close, not to argue from — 'best cell' is a max over five modality sets and the notebook says so in the same breath.",
                ],
                src="notebooks/results.ipynb §1",
            ),
        ],
    ),
    dict(
        id="argument",
        name="How the model was arrived at",
        lead="""The design story, if the audience wants to know why a variant
        tokenizer rather than a bigger CNN. These are measurements that were
        made before <code>locusfusion</code> existed, which is what makes them
        worth showing — they are not a retrospective justification.""",
        items=[
            dict(
                file="results_07_7-where-the-attribution-actually-lands.png",
                chip=None,
                title="Columns carrying 90% of the attribution",
                carries="Ninety percent of what a dense-sequence model reads is a handful of columns out of thousands — and the median isolate differs from H37Rv at only ~14 columns across all 19 loci.",
                notes=[
                    "The pivot. Two independent facts — sparse attribution, near-constant input — say a sequence encoder spends its capacity restating constants.",
                    "Carry its caveat with it: dropping the input SHAP gave 1% of the budget costs ~0.10 AUC, so attribution <em>share</em> mis-ranks value. Only the positional claim survives, and only the positional claim is what the tokenizer is built on.",
                ],
                src="notebooks/results.ipynb §7",
            ),
            dict(
                file="results_09_10-catalogue-additive-what-featurising-a-variant-is-worth.png",
                chip=None,
                title="catalogue → additive, per drug",
                carries="Featurising a variant instead of memorising its identity buys generalisation, and the gain is set almost entirely by how few labels the drug has — worth 0.07 AUC at n = 269.",
                notes=[
                    "The cleanest controlled measurement in the project: one thing changed, identity lookup vs. computed from features.",
                    "The hypothesis named <em>pncA</em>/PYRAZINAMIDE and the answer came back as a small-cohort effect instead. Better result, and it is the one that transfers to a new drug or region.",
                ],
                src="notebooks/results.ipynb §10",
            ),
            dict(
                file="results_06_6-parameters-buy-nothing-above-1m.png",
                chip=None,
                title="Parameters vs. accuracy, established models only",
                carries="The two best established cells are 9× apart in parameters and 0.0015 apart in AUC.",
                notes=[
                    "Rules capacity out before the representation argument starts. Backed by two prior sweeps that closed nothing (<code>joint_capacity</code>, <code>setfusion_scaling</code>: 62 arms, four width axes).",
                    "Redundant with item 1 if you are short on time — item 1 spans both families and four orders of magnitude.",
                ],
                src="notebooks/results.ipynb §6",
            ),
            dict(
                file="results_08_9-the-family-at-both-locus-settings.png",
                chip=None,
                title="The six aggregators, both locus settings",
                carries="None of the six beats the best established model, and none was built to.",
                notes=[
                    "Show it only if you are telling the design story properly. Presented as a leaderboard it reads as 'six things that did not work'; presented as measurements it is where <code>locusfusion</code>'s two design decisions come from.",
                    "The finding worth stating: deleting the gating machinery entirely (<code>deepsets</code>) costs nothing, so the contribution is the representation, not the attention.",
                ],
                src="notebooks/results.ipynb §9",
            ),
        ],
    ),
    dict(
        id="context",
        name="Context and the established models",
        lead="""Part I. What the project measured before the pivot — and the
        only figures that compare against the published baselines on
        input-matched terms.""",
        items=[
            dict(
                file="results_03_3-per-drug-against-both-baselines.png",
                chip=None,
                title="Best established single-drug cell vs. published baselines, by drug",
                carries="Ahead of SD-CNN on all 11 drugs, widest exactly where the baseline had headroom.",
                notes=[
                    "The SD-CNN and MD-CNN lines are the published baselines.",
                    "<code>single-drug, all loci</code> cells cannot be quoted against SD-CNN — different input. They are input-matched to MD-CNN only.",
                ],
                src="notebooks/results.ipynb §3",
            ),
            dict(
                file="results_04_4-what-the-extra-modalities-buy-and-where.png",
                chip=None,
                title="Per-drug modality gain",
                carries="ETHIONAMIDE and ISONIAZID gain from <code>regulatory</code>; PYRAZINAMIDE from <code>protein</code>/<code>biophysical</code>; KANAMYCIN and AMIKACIN gain nothing.",
                notes=[
                    "The gain lands where the resistance mechanism predicts, including the negative control: the <em>rrs</em> drugs target 16S rRNA, which has no protein product to describe.",
                    "This is the project's original question, and the figure that most audiences find most interesting. It is about the established models, not locusfusion.",
                ],
                src="notebooks/results.ipynb §4",
            ),
            dict(
                file="results_05_5-what-the-loci-buy.png",
                chip=None,
                title="What the 19-locus set buys",
                carries="Handing a single-drug model all 19 loci is larger than the modality gain — the cheapest improvement in the project.",
                notes=[
                    "Sets up locusfusion's second claim: the same comparison for the variant-token family is roughly twice this size, because more loci means more tokens rather than more constants.",
                    "KANAMYCIN gains most because its per-drug map is one gene and it was losing the <em>eis</em> promoter. LEVOFLOXACIN's loss is noise at n = 269.",
                ],
                src="notebooks/results.ipynb §5",
            ),
            dict(
                file="results_02_2-the-three-tasks.png",
                chip=None,
                title="Macro CV AUC by task and architecture",
                carries="The 19-locus single-drug panel is the strongest of the three: locus set beats multi-task sharing.",
                notes=[
                    "Every bar is a max over that model's modality sets, so it is not a fair head-to-head — item 2 in the first tier is.",
                    "Also where the transformer trunks (<code>+tf</code>) visibly lose, which is the failure the pivot explains.",
                ],
                src="notebooks/results.ipynb §2",
            ),
        ],
    ),
    dict(
        id="shap",
        name="What the model reads",
        lead="""<code>notebooks/shap_locusfusion.ipynb</code> — GradientSHAP over
        the best CV fold of the <code>sd19</code> cells, gated on re-scoring
        every rebuilt checkpoint against its own run summary. This is the
        interpretability arm, and it is the strongest secondary claim available:
        the model's attribution is exact and names a locus and a column, where
        the established models need SHAP to guess.""",
        items=[
            dict(
                file="shap_lf_06_4-named-variants-the-table-this-whole-pipeline-exists-for.png",
                chip=None,
                title="Named variants against how resistant the allele actually is",
                carries="STREPTOMYCIN keys on <em>rpsL</em> 43R (97% R, 195 isolates) and 88R (97% R); LEVOFLOXACIN on <em>gyrA</em> 94G (100% R). The model was never told any biology.",
                notes=[
                    "The single most persuasive interpretability item — a known resistance mutation arriving at the top of a table the model built itself.",
                    "<b>Read the table before you show it.</b> Several rows are co-resistance rather than mechanism: AMIKACIN's top variant is <em>rpoB</em> 450L, which is a rifampicin mutation marking MDR strains. Present that as the confound it is, or an audience member will.",
                ],
                src="notebooks/shap_locusfusion.ipynb §4",
            ),
            dict(
                file="shap_lf_03_2-which-locus-the-question-these-models-can-actually-fail.png",
                chip=None,
                title="Locus enrichment against the variant-token null",
                carries="Whether the model reads the right gene, scored against the share of the isolate's <em>deviations</em> that fall in that locus rather than against locus length.",
                notes=[
                    "The right null for this architecture, and a stricter one than the conv nets are scored against: <em>rrs</em> is hypervariable, so reading it a lot is not evidence of anything.",
                    "A test the model can fail — all 19 loci are loaded for every drug, so nothing stops it keying pyrazinamide off <em>gyrA</em>.",
                ],
                src="notebooks/shap_locusfusion.ipynb §2",
            ),
            dict(
                file="shap_lf_08_6-per-isolate-does-the-attribution-account-for-the-call.png",
                chip=None,
                title="Summed attribution vs. logit gap, per isolate",
                carries="Completeness — expected gradients decomposes the score exactly, so the points should sit on the diagonal.",
                notes=[
                    "Shows the attribution is not decorative. Off-diagonal points have a concrete meaning here: a pan-wild-type isolate has no variant tokens at all, so its score comes entirely from the <code>[WT]</code> sentinels.",
                    "Show it only to a technical audience; it is a methods slide.",
                ],
                src="notebooks/shap_locusfusion.ipynb §6",
            ),
            dict(
                file="shap_lf_09_7-what-the-tokenizer-actually-saw.png",
                chip=None,
                title="What the tokenizer actually saw",
                carries="The variant census, per locus — the empirical basis for 'a median of ~14 tokens replaces 19 loci × up to 4,066 columns'.",
                notes=[
                    "Pairs naturally with the architecture diagram: this is the input the diagram's first stage consumes.",
                ],
                src="notebooks/shap_locusfusion.ipynb §7",
            ),
        ],
    ),
]

# Items that are not figures.
DOCS = [
    dict(kind="Notebook", path="notebooks/results.ipynb",
         what="""The presentation notebook, and the thing to send if someone
         wants the whole argument. Five parts, executed, 12 figures, every
         caveat written next to the figure it applies to. Built by
         <code>scripts/build_results.py</code> — <b>edit the script, never the
         .ipynb</b>."""),
    dict(kind="Notebook", path="notebooks/shap_locusfusion.ipynb",
         what="""The interpretability arm. Nine figures, the named-variant
         tables per drug, and a §0 gate showing every rebuilt checkpoint was
         re-scored against its own run summary before anything was
         attributed."""),
    dict(kind="Write-up", path="results/experiments/AGGREGATOR_FAMILY_20260827.md",
         what="""The six aggregators: what each measures, what came back, and a
         recommendation on which belong in a write-up. The prose behind the
         second tier."""),
    dict(kind="Write-up", path="results/experiments/CODE_CHANGES_20260825.md",
         what="""The locusfusion design document and the variant census it was
         designed from. The source for the architecture claims."""),
    dict(kind="Notebook", path="notebooks/overview.ipynb",
         what="""The working view — every cell of the task × modality × model
         cube as tables. For answering a question from the floor, not for
         showing."""),
    dict(kind="Figures", path="results/figures/full_run_v2/",
         what="""An older figure set (fig0–figE) from the Part I era, PDF and
         PNG. Superseded by the notebook figures above; keep them only if you
         need a vector version of a Part I panel."""),
]

CAVEATS = [
    ("The claim that survives is <em>matches, at a tenth of the parameters</em>.",
     "+0.0022 paired mean against a fold SD of 0.0172, single seed. <b>Beats</b> is not supported by this data and multi-seeding the headline cells is what would settle it."),
    ("On the multi-drug task it loses, and not narrowly.",
     "0.9118 against <code>late_fusion</code>'s 0.9219. Everything in the first tier is the single-drug task; say which task you are on."),
    ("Held-out test AUC is a tie.",
     "0.9101 against mdcnn's 0.9109 over the same five cells — and both are best-of-5-folds selected, where the baselines use fold 4 unconditionally. Judge on CV."),
    ("Two drugs go the other way, and they go together.",
     "AMIKACIN (−0.014) and KANAMYCIN (−0.020), both <em>rrs</em> drugs. A variant tokenizer over 16S rRNA is the open case."),
    ("The two families are not the same experiment.",
     "Part I reads dense sequence; locusfusion reads reference-difference input. Where they share an axis that is stated, and they are never silently pooled into one ranking."),
    ("Lineage confounding is not controlled anywhere.",
     "<em>M. tuberculosis</em> lineages carry many neutral variants, and a model that learned lineage would look like this one on this split. The variant tokens make it <em>more</em> checkable — attention names the column — but it has not been checked."),
    ("<code>newmodels_full</code> has no README.",
     "The run carrying the headline number is the one run folder in the project without a write-up beside it. Worth fixing before anyone asks how it was launched."),
]


def esc(s):
    return s


def render():
    css = CSS
    parts = [
        "<title>locusfusion display kit</title>",
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&'
        'family=IBM+Plex+Sans:wght@400;500;600&'
        'family=IBM+Plex+Mono:wght@400;500&display=swap">',
        f"<style>{css}</style>",
    ]
    a = parts.append

    a('<header class="masthead">')
    a('<p class="eyebrow">Display kit · exported 2026-09-01</p>')
    a('<h1>What to show, and what each item can carry</h1>')
    a('''<p class="standfirst">Every finished, presentable item in the
       biophysical-fusion project, grouped by the job it does in a talk and
       annotated with the number it actually supports. The theme is that
       <b>locusfusion is the best model we have</b> — the first tier makes that
       case, and the last section is the version of it that survives
       cross-examination.</p>''')
    a('<dl class="keyfacts">')
    for k, v, sub in [
        ("0.9234", "locusfusion mean macro CV AUC", "single-drug, 19 loci, over 5 modality sets"),
        ("0.659M", "parameters", "mdcnn 6.66M · cisfusion 59.3M"),
        ("+0.0022", "paired against mdcnn", "ahead on 4 of 5 modality sets"),
        ("0.9222", "published MD-CNN baseline", "our reproduction 0.9212"),
    ]:
        a(f'<div><dt>{k}</dt><dd>{v}<span>{sub}</span></dd></div>')
    a('</dl>')
    a('<p class="where">Files: <code>results/figures/display/</code> — '
      '23 PNGs and <code>MANIFEST.md</code>, with the regeneration commands.</p>')
    a('</header>')

    a('<nav class="rail" aria-label="Sections">')
    for t in TIERS:
        a(f'<a href="#{t["id"]}"><span>{t["name"]}</span><em>{len(t["items"])}</em></a>')
    a('<a href="#docs"><span>Not a figure</span><em>%d</em></a>' % len(DOCS))
    a('<a href="#caveats" class="warn"><span>Before you present it</span><em>%d</em></a>' % len(CAVEATS))
    a('</nav>')

    a('<main>')
    for t in TIERS:
        a(f'<section id="{t["id"]}" class="tier">')
        a(f'<h2>{t["name"]}</h2>')
        a(f'<p class="lead">{t["lead"]}</p>')
        for it in t["items"]:
            chip = (f'<span class="chip">{it["chip"]}</span>' if it["chip"]
                    else '<span class="chip dot" aria-hidden="true"></span>')
            a('<article class="item">')
            a(f'<figure><img src="{thumb(it["file"])}" alt="{it["title"]}"></figure>')
            a('<div class="meta">')
            a(f'<h3>{chip}{it["title"]}</h3>')
            a(f'<p class="carries">{it["carries"]}</p>')
            a('<ul>')
            for n in it["notes"]:
                a(f'<li>{n}</li>')
            a('</ul>')
            a(f'<p class="src"><code>{it["file"]}</code><span>{it["src"]}</span></p>')
            a('</div></article>')
        a('</section>')

    a('<section id="docs" class="tier">')
    a('<h2>Not a figure</h2>')
    a('<p class="lead">The items that are read rather than projected — send '
      'these when someone wants the argument in full, or the method.</p>')
    a('<div class="docs">')
    for d in DOCS:
        a(f'<div class="doc"><p class="kind">{d["kind"]}</p>'
          f'<p class="path"><code>{d["path"]}</code></p>'
          f'<p class="what">{d["what"]}</p></div>')
    a('</div></section>')

    a('<section id="caveats" class="tier caveats">')
    a('<h2>Before you present it</h2>')
    a('<p class="lead">Each of these is a way the first tier could be wrong, '
      'and they are here because a reviewer finds them anyway. Stating them '
      'first is what makes the claim that <em>does</em> survive land.</p>')
    a('<ol class="cav">')
    for head, body in CAVEATS:
        a(f'<li><p class="ch">{head}</p><p class="cb">{body}</p></li>')
    a('</ol></section>')
    a('</main>')

    a('<footer><p>Figures export from executed notebooks; rebuild with '
      '<code>python scripts/build_results.py &amp;&amp; jupyter nbconvert '
      '--to notebook --execute --inplace notebooks/results.ipynb</code>, then '
      'rerun <code>_build_index.py</code>. Every AUC on this page is 5-fold '
      'CV, macro-averaged over the 11 drugs shared with the BIG-TB '
      'baselines.</p></footer>')
    return "\n".join(parts)


CSS = """
:root{
  --ground:#faf8f4; --panel:#ffffff; --ink:#17140f;
  --ink2:#585149; --ink3:#8a8177; --rule:#e3ded2; --rule2:#efebe2;
  --rust:#a8420f; --rust-soft:#f4e6dd; --blue:#2a6fc4; --amber:#8a5a06;
  --amber-soft:#f7eddb;
  --shadow:0 1px 2px rgba(23,20,15,.05), 0 8px 24px -16px rgba(23,20,15,.24);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#141210; --panel:#1c1917; --ink:#f0ebe2; --ink2:#b3aa9e;
    --ink3:#877e73; --rule:#2f2a25; --rule2:#242019;
    --rust:#e28351; --rust-soft:#33211a; --blue:#7fb0ef; --amber:#d9a441;
    --amber-soft:#2d2415;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --ground:#141210; --panel:#1c1917; --ink:#f0ebe2; --ink2:#b3aa9e;
  --ink3:#877e73; --rule:#2f2a25; --rule2:#242019;
  --rust:#e28351; --rust-soft:#33211a; --blue:#7fb0ef; --amber:#d9a441;
  --amber-soft:#2d2415;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}

*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size:15px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
  padding:0 clamp(18px,5vw,64px) 96px;
}
code{
  font-family:"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  font-size:.86em; background:var(--rule2); color:var(--ink);
  padding:.1em .34em; border-radius:3px; border:1px solid var(--rule);
}
b, strong{font-weight:600}
em{font-style:italic}
a{color:var(--rust)}
:focus-visible{outline:2px solid var(--rust); outline-offset:3px; border-radius:2px}

/* ------------------------------------------------------------- masthead */
.masthead{
  max-width:1120px; margin:0 auto; padding:clamp(48px,7vw,88px) 0 40px;
  border-bottom:1px solid var(--rule);
}
.eyebrow{
  margin:0 0 20px; font-family:"IBM Plex Mono", monospace; font-size:11.5px;
  letter-spacing:.13em; text-transform:uppercase; color:var(--ink3);
}
h1{
  font-family:"Newsreader", Georgia, serif; font-weight:500;
  font-size:clamp(34px,5.2vw,54px); line-height:1.08; letter-spacing:-.015em;
  margin:0 0 22px; text-wrap:balance; max-width:16ch;
}
.standfirst{
  margin:0; max-width:64ch; font-size:17px; line-height:1.66; color:var(--ink2);
}
.standfirst b{color:var(--ink)}
.keyfacts{
  display:grid; gap:1px; margin:40px 0 26px; padding:0;
  grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
  background:var(--rule); border:1px solid var(--rule); border-radius:6px;
  overflow:hidden;
}
.keyfacts > div{background:var(--panel); padding:16px 18px 18px}
.keyfacts dt{
  font-family:"IBM Plex Mono", monospace; font-size:25px; font-weight:500;
  letter-spacing:-.02em; color:var(--rust); font-variant-numeric:tabular-nums;
}
.keyfacts dd{
  margin:4px 0 0; font-size:13px; line-height:1.42; color:var(--ink);
}
.keyfacts dd span{display:block; color:var(--ink3); font-size:12px; margin-top:3px}
.where{margin:0; font-size:13.5px; color:var(--ink3)}

/* ----------------------------------------------------------------- rail */
.rail{
  position:sticky; top:0; z-index:5; max-width:1120px; margin:0 auto;
  display:flex; flex-wrap:wrap; gap:2px; padding:10px 0;
  background:var(--ground); border-bottom:1px solid var(--rule);
}
.rail a{
  display:flex; align-items:baseline; gap:7px; text-decoration:none;
  padding:5px 11px; border-radius:4px; font-size:13px; color:var(--ink2);
}
.rail a:hover{background:var(--rule2); color:var(--ink)}
.rail a em{
  font-style:normal; font-family:"IBM Plex Mono",monospace; font-size:11px;
  color:var(--ink3);
}
.rail a.warn{color:var(--amber)}
.rail a.warn em{color:var(--amber)}

/* ---------------------------------------------------------------- tiers */
main{max-width:1120px; margin:0 auto}
.tier{padding:56px 0 8px; border-bottom:1px solid var(--rule)}
.tier:last-child{border-bottom:0}
h2{
  font-family:"Newsreader", Georgia, serif; font-weight:500; font-size:29px;
  letter-spacing:-.01em; margin:0 0 12px; text-wrap:balance;
}
.lead{margin:0 0 34px; max-width:66ch; color:var(--ink2); font-size:14.5px}

.item{
  display:grid; grid-template-columns:minmax(0,340px) minmax(0,1fr);
  gap:clamp(20px,3vw,38px); align-items:start;
  padding:26px 0; border-top:1px solid var(--rule2);
}
.item:first-of-type{border-top:0; padding-top:6px}
.item figure{margin:0}
.item img{
  display:block; width:100%; height:auto; border-radius:5px;
  border:1px solid var(--rule); background:#fcfcfb; box-shadow:var(--shadow);
}
.meta h3{
  display:flex; align-items:center; gap:11px; margin:0 0 10px;
  font-family:"Newsreader", Georgia, serif; font-weight:600; font-size:20px;
  letter-spacing:-.005em; line-height:1.25; text-wrap:balance;
}
.chip{
  flex:none; width:24px; height:24px; border-radius:50%;
  display:grid; place-items:center;
  background:var(--rust); color:var(--ground);
  font-family:"IBM Plex Mono",monospace; font-size:12px; font-weight:500;
}
.chip.dot{
  width:7px; height:7px; background:var(--rust); opacity:.55; margin-left:8px;
}
.carries{
  margin:0 0 12px; font-size:15px; line-height:1.58;
  padding-left:14px; border-left:2px solid var(--rust);
}
.meta ul{margin:0 0 14px; padding-left:18px; color:var(--ink2); font-size:14px}
.meta li{margin:0 0 6px}
.meta li b{color:var(--ink)}
.src{
  margin:0; display:flex; flex-wrap:wrap; gap:10px; align-items:baseline;
  font-size:12px; color:var(--ink3);
}
.src code{font-size:11.5px; background:transparent; border:0; padding:0; color:var(--ink3)}

/* ----------------------------------------------------------------- docs */
.docs{
  display:grid; gap:1px; background:var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(268px,1fr));
  border:1px solid var(--rule); border-radius:6px; overflow:hidden;
  margin-bottom:16px;
}
.doc{background:var(--panel); padding:18px 20px 20px}
.kind{
  margin:0 0 8px; font-family:"IBM Plex Mono",monospace; font-size:10.5px;
  letter-spacing:.12em; text-transform:uppercase; color:var(--rust);
}
.path{margin:0 0 9px}
.path code{font-size:12.5px; word-break:break-all}
.what{margin:0; font-size:13.5px; line-height:1.55; color:var(--ink2)}

/* -------------------------------------------------------------- caveats */
.caveats h2{color:var(--amber)}
.cav{margin:0; padding:0; list-style:none; counter-reset:c; max-width:78ch}
.cav li{
  counter-increment:c; position:relative; padding:16px 0 16px 42px;
  border-top:1px solid var(--rule2);
}
.cav li:first-child{border-top:0}
.cav li::before{
  content:counter(c); position:absolute; left:0; top:17px;
  font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--amber);
  background:var(--amber-soft); width:24px; height:24px; border-radius:4px;
  display:grid; place-items:center;
}
.ch{margin:0 0 5px; font-size:15px; font-weight:500}
.cb{margin:0; font-size:14px; color:var(--ink2)}

footer{
  max-width:1120px; margin:56px auto 0; padding-top:22px;
  border-top:1px solid var(--rule); font-size:12.5px; color:var(--ink3);
}
footer p{margin:0; max-width:78ch}

@media (max-width:760px){
  .item{grid-template-columns:1fr; gap:16px}
  .rail{position:static}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important}}
"""

OUT.write_text(render(), encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
