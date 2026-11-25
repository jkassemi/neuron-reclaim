// Full Phase-2 Paper — Zero-Gradient Surgical Concept Grafting (VibeSplice v0.4.0)
// Clean modern academic typography (Libertinus Serif, generous leading & measure)

#let navy = rgb("#1B365D")
#let teal = rgb("#0E7C7B")
#let slate = rgb("#475569")
#let ink = rgb("#111827")
#let lightbg = rgb("#F8FAFC")
#let rulecol = rgb("#CBD5E1")

#set page(
  paper: "us-letter",
  margin: (left: 0.98in, right: 0.98in, top: 0.88in, bottom: 0.88in),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(font: "Libertinus Serif", size: 8.2pt, fill: luma(110))
      #grid(
        columns: (1fr, auto),
        [Zero-Gradient Surgical Concept Grafting (SCG)],
        [Kassemi (2026)],
      )
      #v(-0.42em)
      #line(length: 100%, stroke: 0.4pt + rulecol)
    ]
  },
  footer: context [
    #set text(font: "Libertinus Serif", size: 8.5pt, fill: luma(110))
    #h(1fr) #counter(page).display() #h(1fr)
  ],
)

#set text(font: "Libertinus Serif", size: 10pt, fill: ink, lang: "en", hyphenate: true)
#show raw: set text(font: "DejaVu Sans Mono", size: 0.82em, fill: rgb("#1E293B"))
#set par(justify: true, leading: 0.68em, spacing: 1.30em, first-line-indent: 0pt)
#set list(spacing: 0.65em, body-indent: 0.55em)
#set enum(spacing: 0.65em, body-indent: 0.55em)
#set heading(numbering: "1.1.")

#show heading.where(level: 1): it => {
  block(above: 1.9em, below: 0.85em, sticky: true)[
    #set text(size: 12pt, weight: "bold", fill: navy, hyphenate: false)
    #it
  ]
}
#show heading.where(level: 2): it => {
  block(above: 1.45em, below: 0.65em, sticky: true)[
    #set text(size: 10.4pt, weight: "bold", fill: rgb("#1E293B"), hyphenate: false)
    #it
  ]
}
#show figure: it => {
  block(above: 1.2em, below: 1.35em, width: 100%, it)
}
#show figure.caption: it => {
  set text(size: 8.6pt, fill: luma(55))
  set par(leading: 0.56em)
  v(0.35em)
  block(width: 98%, align(left, it))
}
#set table(
  stroke: none,
  inset: (x: 3.4pt, y: 3.3pt),
  align: (col, row) => if col == 0 { left } else { right },
)
#show table: set text(size: 8.2pt, number-width: "tabular")

#let propbox(title, body) = block(
  above: 1.35em,
  below: 1.45em,
  width: 100%,
  fill: lightbg,
  inset: (x: 13pt, y: 11pt),
  radius: 3.5pt,
  stroke: (left: 2.5pt + navy, top: 0.4pt + rulecol, right: 0.4pt + rulecol, bottom: 0.4pt + rulecol),
  breakable: false,
)[
  #set par(leading: 0.66em, spacing: 0.72em)
  #set list(spacing: 0.68em)
  #set enum(spacing: 0.68em)
  #text(size: 9.5pt, weight: "bold", fill: navy)[#title] #h(0.3em)
  #text(size: 9.4pt)[#body]
]

// ==================== TITLE BLOCK ====================
#line(length: 100%, stroke: 1.8pt + navy)
#v(0.75em)
#align(center)[
  #set par(justify: false, leading: 0.52em)
  #set text(hyphenate: false)
  #text(size: 15.6pt, weight: "bold", fill: rgb("#0F172A"))[
    Zero-Gradient Surgical Concept Grafting: Scaling, Cross-Architecture Generalization, Transactional Rollback, and #box[Weight-Space] Auditing
  ]
  #v(0.7em)
  #text(size: 10.8pt, weight: "bold")[James Kassemi]
  #v(0.22em)
  #text(size: 9.0pt, style: "italic", fill: navy)[Developed in close collaboration with Gemini 4 Neon]
  #v(0.28em)
  #text(size: 8.8pt, fill: slate)[
    Independent Researcher #h(0.55em) · #h(0.55em) `james@kassemi.org` #h(0.55em) · #h(0.55em) ORCID: 0009-0007-0438-6421 \
    Code (VibeSplice v0.4.1, MIT): `github.com/jkassemi/vibesplice` #h(0.55em) · #h(0.55em) DOI: 10.5281/zenodo.24544499
  ]
]
#v(0.55em)
#line(length: 100%, stroke: 0.6pt + rulecol)
#v(0.7em)

// ==================== ABSTRACT ====================
#align(center)[
  #block(width: 98%, fill: lightbg, inset: (x: 13pt, y: 11pt), radius: 3.5pt, stroke: 0.45pt + rulecol)[
    #align(left)[
      #set text(size: 9.3pt)
      #set par(leading: 0.65em, spacing: 1.15em)

      #text(weight: "bold", fill: navy)[Abstract.] #h(0.25em)
      Teaching a frozen language model new concepts typically forces a compromise between gradient fine-tuning—which is data-hungry, opaque, and prone to collateral forgetting—and in-context prompting, which vanishes once the context resets. We present *Surgical Concept Grafting (SCG)*, a closed-form, zero-gradient method that installs permanent, composable concepts directly into pretrained weights using just *one unused vocabulary slot and four silent MLP neurons per novel concept* (0.00083% of a 4B model, 66.6 KB in BF16), with no training corpus, no runtime hooks, and standard HuggingFace checkpoint portability.

      Rather than approximating edits with gradient steps, SCG compiles knowledge directly into a transformer's native geometry: separating input recognition from output addressing, routing multi-scale attributes through reclaimed silent neurons so unmodified upper layers resolve zero-shot multi-hop queries, and jointly orthogonalizing concurrent concepts to prevent cross-talk. On Qwen3-4B, a *100-concept graft* achieves *92.0% concept-level Top-1* (92/100 concepts succeeding on at least one of two unseen paraphrases: 43 on both, 49 on one, 8 on neither) and *67.5% prompt-level Top-1* (135/200 held-out prompts), lifting multi-hop Top-10 accuracy to 20% versus 11% before editing while holding KL divergence to 0.0070 and 0.0098 nats on 10-prompt general English and 10-prompt adversarial near-neighbor suites, respectively. With zero per-model tuning, the same recipe replicates on a 5-concept pack across four tied and untied families (Qwen3-4B, Qwen2.5-7B, Llama-3.2-3B, Nanbeige4.2-3B) plus a 20-concept non-CJK Llama-3.2-3B stack, outperforming matched-budget AdamW and 3×-larger LoRA on joint multi-hop recall, trap specificity, and collateral preservation.

      Because every novel concept maps to an exact, tiny weight address, SCG turns static checkpoints into transactional knowledge stores—supporting 10.6 s in-place updates, 0.88 ms bit-exact rollback of edited parameter slots, an initial multi-token verbatim fact pilot (zero vocabulary edits, one silent neuron per output token; 3/10 exact unseen completions including a 13-token phone number), and a companion weight-space auditor achieving 1.0000 AUROC across all nine tested single- and split-neuron covert injection regimes.
    ]
  ]
]

#v(0.35em)

= Introduction

Recent work in *weight-space programming* demonstrates that pretrained neural networks are not opaque black boxes editable only by gradient descent: their weights can be read, compiled, and rewritten directly [5, 6, 7], extending classical key–value model editing such as ROME and MEMIT [1, 2] and geometric studies of linear and non-linear feature structure [3, 4]. Yet turning this observation into a practical mechanism for *installing, updating, and erasing new concepts* inside frozen language models—at scale, across model families, and with minimal collateral drift—remains an open challenge.

In our companion technical note [9], we introduced a minimal closed-form proof-of-concept installing $K=2$ sensory concepts into Qwen3-4B. Here we resolve its open scaling, cross-architecture, baseline, continuous-learning, and dual-use security questions in full, implemented in the open-source MIT toolkit *VibeSplice* (v0.4.1) [10]. Specifically, we contribute:

- *Analytical refinements for high-$K$ capacity:* Linear cone-gated SwiGLU (`w_up = b`), exact QR Write orthogonalization, capped closed-form orthogonal-residual RMSNorm calibration, activation-filtered quiet slot reclamation (eliminating CJK reliance), an *Untied Pure-Orthogonal Theorem* ($sqrt(2)times$ Write gain), and *Leave-One-Out Joint Anti-Bleed Compilation* scaling on Qwen3-4B to $K = 50$–$100$ concepts ($< 0.01$ English/adversarial KL, 90–92% concept-level / 67.5–75.0% prompt-level Top-1).
- *Four-family cross-architecture validation and matched-budget baselines:* Zero-tuning 5-concept replication across Qwen3-4B, Qwen2.5-7B, Meta Llama-3.2-3B-Instruct (plus $K=20$ non-CJK), and Nanbeige4.2-3B (with an honest boundary diagnostic on narrow Qwen2.5-3B), alongside a learning-rate and step sweep comparing SCG against matched-budget sparse AdamW and 3×-budget LoRA ($r=4$).
- *Transactional continuous learning, verbatim multi-token chains, and stealth-graft auditing:* In-place concept mutation (10.6 s), bit-exact rollback of edited parameter slots (0.88 ms, 0.000000 printed Write probability), a five-fact multi-token verbatim pilot for _Due West, South Carolina_ (0 embedding edits; 3/10 unseen exact), and a Red-Team vs. Blue-Team audit establishing a norm–collinearity trade-off achieving 1.0000 AUROC across all nine tested single- and split-neuron attack configurations ($M <= 32$).

= Geometric Primitives & Closed-Form SCG

*Empirical motivation (kind twists live downstream of Layer 0).* Across 11 semantic domains and 16 perceptual modifiers (_bright, sharp, warm, rich, ..._), raw CKA (0.86–0.93) suffers an $approx 0.70$ small-$N$/high-$D$ permutation floor, and bare modifier words explain 30–48% of offset variance. Factoring out additive word strength via rank-1 log-factorization ($log c[A,B] = u_A + u_B + "excess"_(A,B)$, 5,000 permutation shuffles) reveals significant non-additive *kind twists* shared within `CHEMO` = {_taste, smell_}, `AUDIO` = {_sound, voice, music_}, and `MIND` = {_emotion, memory_} across Layers 6–36 of Qwen3-4B ($T = +0.088 -> +0.193$, L24 $p = 0.0006$; zero at L0).

Crucially, passive in-context binding (_"A daxen is a kind of flavor..."_) fails pre-registered binding gates (4/8, margin −1.1%), whereas *grafting one Layer-0 embedding row* into an unused vocabulary slot causes Layers 6–36 to dynamically compute the exact kind-specific twist (`CHEMO` +0.742, `AUDIO` +0.655, random +0.040) and produces a 10/10 open-ended behavioral double dissociation ($B = +2.999$ vs. $-2.968$). To extend this read-only kind graft into a composable, high-$K$ *Read + Multi-Hop + Write* concept without backpropagation, SCG edits *1 vocabulary slot and 4 quiet SwiGLU neurons* ($13D$ floats per concept on tied models; $14D$ on untied models).

== Stage 1: RMSNorm–Head Scale Asymmetry & Untied Pure-Orthogonal Addressing

Naively placing $bold(e)_G = r_"ref" hat(bold(e))_K$ into a tied embedding table (`embed_tokens == lm_head`) leaks 77.7% probability to $chevron.l G chevron.r$ on near-neighbor prompts (_"The soup had a rich and savory"_, adversarial KL = 0.413 nats).

#propbox[Proposition 1 (Scale Asymmetry & Untied Pure-Orthogonal Theorem).][
  Let $hat(bold(e))_K$ be the unit parent-kind centroid and let $hat(bold(u))_perp$ be a unit *private address* QR-orthogonalized against all parent kinds and prior grafts.
  - *Tied heads (`embed_tokens == lm_head`):* Setting $bold(e)_G = gamma r_"in" (cos theta hat(bold(e))_K + sin theta hat(bold(u))_perp)$ exploits Layer-0 RMSNorm invariance to $gamma > 0$ on the input side while exponentially suppressing unwanted output softmax probability as $exp(-(1-gamma)z)$ (4,503× KL cut at $gamma = 0.50$; 275,496× at $gamma = 0.25$; Figure 1a).
  - *Untied heads (`embed_tokens != lm_head`):* Decoupling the input row at full norm ($bold(e)_("in",G) = r_"in" (cos 30 degree hat(bold(e))_K + sin 30 degree hat(bold(u))_perp)$) from a *purely orthogonal* output row ($bold(e)_("out",G) = gamma r_"out" hat(bold(u))_perp$) removes all parent-kind overlap in `lm_head` (AdvMaxG = 0.000000) while increasing private-address Write gain by $sin(90 degree) / sin(45 degree) = sqrt(2) approx 1.414times$, reducing required Write `down_proj` norm by $1 - 1/sqrt(2) approx 29.3%$.
]

== Stage 2: Activation-Filtered Quiet Reclamation & Multi-Scale Linear Read

*Architecture-agnostic quiet slot reclamation.* Selecting neurons by minimum `down_proj` norm harms English (0.00557 KL) by grabbing low-norm active English neurons; pure static CJK projection cuts single-concept KL 8.2× (0.00060) but still lets polysemantic bridge neurons slip in by $K=17$ (AdvKL cliff $0.002 -> 0.012$) and assumes a CJK-heavy vocabulary.

Instead, we run *one zero-corpus forward pass* over 8 generic English whitening sentences (`_DEFAULT_WHITEN_GEN`, disjoint from all evaluation suites) to measure each neuron's worst-case English residual perturbation $E_j = max_t |"SiLU"(bold(w)_("gate",j)^top bold(x)_t)(bold(w)_("up",j)^top bold(x)_t)| dot ||bold(w)_("down",:,j)||_2$, ranking CJK-leaning neurons (or all neurons on English-centric models like Llama-3) by ascending $E_j$. This eliminates the $K=17$ cliff (AdvKL cut 2.6× at $K=50$) and achieves GenKL = 0.00054 on Llama-3.2-3B-Instruct with zero CJK reliance.

*Multi-scale linear attribute superposition.* Causal activation patching confirms that two-token adjective–noun deltas (_"The Icelandic flavor"_ − _"The common flavor"_) transfer 0.0% across L2–L20 because attention reads _"Icelandic"_ from its own token position rather than compressing it into _"flavor"_. Because attention projections are linear after RMSNorm, superposing the attribute token's _own_ position-matched centered state onto $chevron.l G chevron.r$ across three proportional depths (6%, 33%, 50% of layers—e.g. L2, L12, L18 in 36-layer models: $alpha_1 bold(d)_"adj"$, $alpha_2 (0.5 bold(d)_"adj" + 0.5 bold(d)_"noun")$, $alpha_3 bold(d)_"noun"$) lets unmodified upper layers resolve direct and zero-shot multi-hop queries (_capital city, spoken language, package manager_) without seeing a single origin prompt during fitting. As shown in Figure 1b (Exp 4), setting #box[`read_alphas = [0.8, 1.0, 0.8]`] resolves the v1 trade-off between multi-hop recall and parent-kind separation, hitting [\#1, \#1] on _Reykjavik_ and _Tokyo_ while lifting BehavSep 3.8× (+0.47 $->$ +1.79) at GenKL = 0.00043.

#figure(
  image("figs/fig1_mechanism_and_pareto.svg", width: 100%),
  caption: [*(a) RMSNorm–tied-head scale asymmetry* (Qwen3-4B, 1-row `CHEMO` graft): shrinking $gamma$ keeps 81–100% downstream twist and behavioral dissociation while suppressing adversarial near-neighbor KL by 4,503×–275,496×. *(b) Read-alpha Pareto sweep (Exp 4):* reducing early L2 superposition from $1.5 -> 0.8$ (`[0.8, 1.0, 0.8]`) preserves Rank \#1 on both _Reykjavik_ (9.3%) and _Tokyo_ (21.9%) while lifting parent-kind double dissociation $Delta B$ from +0.47 to +1.79 nats (GenKL = 0.00043).]
)

== Stage 3: Linear Cone-Gated Write, Exact QR Projection & Joint Anti-Bleed

Modern SwiGLU MLPs lack bias terms. In v1, setting $bold(w)_"gate" = s(bold(k) - tau bold(b))$ and $bold(w)_"up" = bold(k)$—where $bold(b)$ is the ridge-solved cone unit-response vector satisfying $bold(b)^top bold(x) approx 1$ across residual activations—yielded high fit SNR (669×–29,475×), but suffered *quadratic activation sensitivity* ($prop (bold(k)^top bold(x))^2$): unseen paraphrases with $bold(k)^top bold(x) = 0.85$ collapsed to 35–58% of calibrated activation while outliers with $bold(k)^top bold(x) = 2.0$ exploded 5×. Furthermore, sequential non-orthonormalized subtraction of $bold(w)_"norm" dot.o bold(p)_k$ drifted to 9.8% geometric cross-talk by $K=50$, and uncapped margin calibration against already-edited logits triggered an exponential norm arms-race (72 $->$ 1,608) above $K=50$.

#propbox[Proposition 2 (Linear Cone-Gating, RMSNorm Asymptotic Ceiling & Joint Anti-Bleed).][
  1. *Linear cone-gated SwiGLU:* Setting $bold(w)_"up" = bold(b)$ (instead of $bold(k)$) with $bold(w)_"gate" = s(bold(k) - tau bold(b))$ ($tau = 0.55, s = 12.0$) gives $"act"(bold(x)) approx "SiLU"(s(bold(k)^top bold(x) - tau)) dot 1$, linear above threshold and $approx -0.009$ on negatives.
  2. *QR pre-RMSNorm orthogonalization & orthogonal-residual RMSNorm calibration:* Let $hat(bold(d))_("write",i)$ be the two-pass QR projection of $bold(w)_"norm" dot.o hat(bold(u))_(perp,i)$ orthogonal to $"span"({bold(w)_"norm" dot.o hat(bold(e))_(K_m)} union {bold(w)_"norm" dot.o hat(bold(u))_(perp,j) : j != i})$, canceling all *pre-RMSNorm linear logit projections* onto parent kinds and other grafted addresses ($chevron.l hat(bold(d))_("write",i), bold(w)_"norm" dot.o bold(e)_("out",j) chevron.r = 0$ for $j != i$; any residual final-logit coupling via shared RMSNorm denominator scaling is measured end-to-end by the cross-bleed matrix). Adding $m hat(bold(d))_("write",i)$ to pre-norm residual $bold(h)$ (RMS $overline(rho)$) yields new RMS $rho' = sqrt(overline(rho)^2 + (2 m / D) chevron.l bold(h) "," hat(bold(d))_("write",i) chevron.r + m^2 / D)$ and rescales the baseline target logit $z_0$ by $overline(rho) / rho'$. Under the high-dimensional *orthogonal-residual approximation*—valid when $chevron.l bold(h), hat(bold(d))_("write",i) chevron.r approx 0$ and the unedited logit $z_0 approx 0$ for a private orthogonal token prior to firing—both cross- and rescaling terms vanish, giving asymptotic ceiling $z_"max" = sqrt(D) g_"eff"$ ($g_"eff" = chevron.l hat(bold(d))_("write",i), bold(w)_"norm" dot.o bold(e)_("out",i) chevron.r$) and closed-form calibration for $Delta z = min(26, 0.78 z_"max", max(20, z_"gap,vocab" + 5))$ (excluding grafted tokens from $z_"gap,vocab"$):
     $ m_"req" = (Delta z dot overline(rho)) / sqrt(max(g_"eff"^2 - Delta z^2 / D, 0.25 g_"eff"^2)), quad bold(w)_("down",i) = (m_"req" / a_("ref",35%)) hat(bold(d))_("write",i). $
  3. *Leave-One-Out Joint Anti-Bleed Whitening:* When compiling a pack of $K$ concepts from 8 syntactically diverse naming templates each, we precompute $bold(S)_"pack" = sum_(j=1)^K bold(X)_j^top bold(X)_j$ once and solve each concept's key against $bold(C)_(-,i) = bold(C)_"bg" + 1.5 bold(C)_("hard",i) + 3.0 (bold(S)_"pack" - bold(X)_i^top bold(X)_i) / (8(K-1))$ in $O(8 D^2)$ time—zeroing out shared hypernym directions (_coffee shop_, _bakery_) across co-installed concepts with zero evaluation leakage.
]

= Scaling to $K = 100$ Simultaneous Concepts (Experiment 1)

We benchmark $K = 1 -> 100$ simultaneously grafted novel concepts on Qwen3-4B-Instruct-2507 (267 free in-bounds padding slots), balanced across five categories of 20 concepts each: *place* (country flavors/sounds $->$ capital city), *city* (U.S. city local businesses, intentionally reusing _coffee shop_ and _bakery_ across cities $->$ U.S. state), *cuisine* (regional dishes $->$ language/country), *language* (dev tools $->$ ecosystem/package manager), and *product* (consumer brands $->$ activity).

Every concept is evaluated on 2 strictly unseen Write paraphrases (0% template overlap), reporting both *prompt-level Top-1* (across all $2K$ prompts) and *concept-level Top-1* (succeeding on at least one of a concept's two unseen paraphrases). We additionally evaluate 1 in-stack wrong-attribute trap per concept, 1 zero-shot multi-hop Read prompt (vs. unmodified baseline), the full $K times (K-1)$ cross-concept bleed matrix, 20 disjoint English KL prompts (10 general prose, 10 adversarial sensory near-neighbors), and L35 `down_proj` norm audit Precision\@K.

#figure(
  image("figs/fig2_scaling_1_to_100.svg", width: 100%),
  caption: [*Scaling $K = 1 -> 100$ simultaneous concepts on Qwen3-4B-Instruct-2507 (Exp 1).* *(a)* Unmodified v1 plateaus at 40% prompt / 56% concept Top-1 due to quadratic SwiGLU collapse and 4-shot overfitting; v3b linear cone-gating lifts recall to 82–90% through $K=50$, and *v4 Joint Anti-Bleed* reaches 90.0% concept Top-1 at $K=50$ (0 / 4,900 bleed $> 1%$) and *92.0% at $K=100$* (0.083% of weights). *(b)* Quiet-CJK reclamation removes v1's $K=17$ polysemantic bridge cliff (AdvKL $0.013 -> 0.0036$ at $K=50$) and holds both GenKL (0.0070) and AdvKL (0.0098) below 0.01 nats all the way to $K=100$.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    table.hline(stroke: 0.9pt),
    table.header(
      [*Variant ($K$)*], [*Edited*], [*Write (P/C)*], [*X-Bleed (p99/Max)*], [*Trap*], [*MH Top-10 (Base)*], [*Gen KL*], [*Adv KL*], [*GeomOff*]
    ),
    table.hline(stroke: 0.5pt),
    [v1 base (10)], [0.008%], [40.0 / 60.0%], [.00001 / .00002], [.0958], [90% (20%)], [.00084], [.00173], [.0753],
    [v1 base (50)], [0.041%], [40.0 / 56.0%], [.00006 / .9986], [.0219], [64% (12%)], [.00178], [.01297], [.0980],
    table.hline(stroke: 0.35pt + rulecol),
    [v3b seq. (20)], [0.017%], [77.5 / 90.0%], [.00000 / .9796], [.0501], [80% (15%)], [.00127], [.00200], [.0573],
    [v3b seq. (50)], [0.041%], [64.0 / 82.0%], [.00001 / 1.000], [.0200], [60% (14%)], [.00257], [.00356], [.0636],
    [v3b seq. (100)], [0.083%], [49.0 / 67.0%], [.00001 / 1.000], [.0241], [34% (11%)], [.00569], [.00769], [.0636],
    table.hline(stroke: 0.35pt + rulecol),
    [*v4 Joint (50)*], [0.041%], [*75.0 / 90.0%*], [*.00002 / .00119*], [.0208], [34% (12%)#super[†]], [*.00216*], [.00643], [*.0000000*],
    [*v4 Joint (100)*], [0.083%], [*67.5 / 92.0%*], [*.00004 / .9996*], [.0522], [20% (11%)#super[†]], [*.00703*], [*.00983*], [*.0000001*],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Scaling progression on Qwen3-4B-Instruct-2507 across identical unseen evaluation suites (P/C = Prompt-level Top-1 over $2K$ prompts / Concept-level Top-1 succeeding on $>= 1$ of 2 unseen paraphrases; Audit Precision\@K = 1.00 in every row; KL measured on 10 general English and 10 adversarial near-neighbor prompts). #super[†]v4 uses `read_alphas = [0.8, 1.0, 0.8]` vs. v3b's `[1.5, 1.2, 0.8]`; one-sided exact binomial sign tests on continuous per-concept multi-hop log-probability lift ($Delta log p = log p_"graft" - log p_"base" > 0$, non-positive ties counted against improvement) confirm positive transfer at both $K=50$ (42/50 positive, mean lift $+1.87$ nats, $p = 5.8 times 10^(-7)$) and $K=100$ (73/100 positive, mean lift $+0.65$ nats, $p = 2.3 times 10^(-6)$). Paraphrase breakdown for v4: at $K=50$, 30 concepts hit Top-1 on both paraphrases, 15 on one, and 5 on neither (40/40 non-product concepts hit $>= 1$ paraphrase, 0 / 4,900 pairs $> 1%$ bleed); at $K=100$, 43 hit both, 49 hit one, and 8 hit neither (99.95% clean, 9 / 19,800 $> 1%$).]
)

*Diagnostic takeaways.* Three empirical bugs surfaced and resolved cleanly during scaling: (1) v1's $K=17$ AdvKL jump (0.002 $->$ 0.012) was caused by a single mis-ranked English bridge neuron zeroed before grafting—completely removed by quiet activation filtering; (2) uncapped margin calibration against edited logits caused an exponential norm cascade (72 $->$ 1,608 at $K=100$, AdvKL = 0.182)—clamping against pretrained vocab cut $K=100$ AdvKL 23.6× (0.182 $->$ 0.0077) with max $||bold(w)_"down"|| = 108.3$; and (3) joint leave-one-out whitening eliminated sequential cross-bleed at $K=50$ (max 1.000 $->$ 0.00119, an 840× drop) and lifted $K=100$ _city_ concept Top-1 from 45% $->$ 95% while driving geometric cross-talk to $10^(-7)$.

= Cross-Architecture Generalization & Matched-Budget Baselines

== Four-Family Generalization & Untied Pure-Orthogonal Verification (Experiment 2)

Using zero per-model hyperparameter tuning (depth-proportional schedule `[6%, 33%, 50%, L-1]`), we evaluate the 5-concept cross-domain pack (_solis, kora, HarborRoast, ZestCake, IronStore_) across four distinct models—plus an A/B test of Proposition 1's Untied Pure-Orthogonal Theorem on both untied checkpoints (Qwen2.5-7B and Nanbeige4.2-3B, verifying the theorem both inside and outside Qwen) and a $K=20$ non-CJK stack on Meta Llama-3.2-3B-Instruct (allocating Meta's official unused `<|reserved_special_token_*|>` slots).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    table.hline(stroke: 0.9pt),
    table.header(
      [*Model & Mode*], [*Arch / T*], [*Write (P/C)*], [*XBleed*], [*Trap*], [*MH Ranks (Base)*], [*$Delta B$*], [*Gen / Adv KL*], [*Gain / $||w_d||$*]
    ),
    table.hline(stroke: 0.5pt),
    [Qwen3-4B (K=5)], [36L, 2560, T], [80 / 100%], [.00000], [1.00#super[‡]], [100% top-10], [+1.79], [.0007 / .0011], [— / 70.6],
    [Qwen2.5-7B (tied)], [28L, 3584, U], [*100 / 100%*], [.00003], [.00000], [[1,1,13,4,1] (+4.04)], [+3.55], [.0056 / .0034], [1.10 / 63.8],
    [*Qwen2.5-7B (untied\*)*], [28L, 3584, U], [*100 / 100%*], [*.00001*], [*.00000*], [*[1,1,13,10,2]*], [+3.16], [.0052 / .0085], [*1.55 / 45.5*],
    [Nanbeige-3B (tied)], [22L, 3072, U], [90 / 100%], [.00000], [.00000], [[2,1,67,6,59]], [+2.75], [*.0010 / .0041*], [1.77 / 60.3],
    [*Nanbeige-3B (untied\*)*], [22L, 3072, U], [*100 / 100%*], [*.00000*], [.9930], [*[1,1,17,4,13]*], [+2.35], [.0014 / .0047], [*2.50 / 42.6*],
    [*Llama-3.2 (K=5)*], [28L, 3072, T], [*90 / 100%*], [*.00002*], [*.00020*], [*[1,1,10,9,65]* (+1.88)], [*+2.46*], [*.0005 / .0006*], [— / 44.1],
    [*Llama-3.2 (K=20)*], [28L, 3072, T], [*82.5 / 100%*], [*.00008*], [*.00047*], [25% (25%, +0.32)], [*+2.49*], [*.0014 / .0011*], [— / 46.2],
    [Qwen2.5-3B (bound.)], [36L, 2048, T], [80 / 100%], [.00000], [.0115], [flat (0%)], [+0.03#super[§]], [.0027 / .0023], [1.61 / 32.2],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Zero-tuning cross-architecture generalization (Exp 2; Audit P\@K = 1.00 on every model). On both untied models (Qwen2.5-7B and SentencePiece 22-layer Nanbeige4.2-3B), `untied*` confirms Proposition 1 to 4 decimal places: Write gain scales by *exactly $sqrt(2) = 1.414times$* (1.099 $->$ 1.554, 1.766 $->$ 2.498), $||bold(w)_"down"||$ drops *-29%*, and AdvMaxG = 0.000000. #super[‡]Single Portland$->$Seattle Pacific-NW twin trap. #super[§]Diagnostic confirms Qwen2.5-3B retains full 1-row kind inheritance at $theta = 0 degree, gamma = 1.0$ ($Delta B = +3.34$), isolating the flat Read to excessive rotation ($theta = 45 degree$) at narrower $D = 2048$.]
)

*Eliminating the CJK-surplus objection.* On English-centric Meta Llama-3.2-3B-Instruct, pure English-energy quiet reclamation achieves GenKL = 0.00054 at $K=5$ (lower than CJK reclamation on Qwen3-4B), lifts _Reykjavik_ from baseline Rank \#251 $->$ \#1 and _Tokyo_ \#8 $->$ \#1, maintains BehavSep = +2.49, and scales cleanly to $K=20$ (20/20 concepts at Rank \#1, XBleedMax = 0.00008, TrapMax = 0.00047, GenKL = 0.00140).

== Matched-Budget AdamW & LoRA Baselines (Experiment 3)

We compare closed-form SCG directly against gradient fine-tuning on the exact same dual-domain benchmark (_solis_ + _kora_, Qwen3-4B). To ensure a generous baseline, every gradient run starts from SCG's kind-base embedding initialization and receives *16 supervised prompt$->$target pairs per concept*—including explicit _"This solis originates in Iceland"_ pairs that SCG never sees—never mentioning _Reykjavik_, _Tokyo_, or any evaluation paraphrase.

We sweep learning rates and steps for (i) *GradSparse-CE* and (ii) *GradSparse-KL* (KL-anchored on `_DEFAULT_WHITEN_GEN`) updating the *exact same 66,560 floats*, and (iii) *LoRA $r=4$ + KL* on L2, L12, L18, L35 `down_proj` + `embed[G]` (201,728 floats, 3× SCG's budget).

#figure(
  image("figs/fig3_baselines_and_crossarch.svg", width: 100%),
  caption: [*(a) SCG vs. matched-budget AdamW and 3×-budget LoRA $r=4$ (Exp 3).* Closed-form SCG (`α = [0.8, 1.0, 0.8]`, 6.2 s, 0 backprop steps) uniquely occupies the top-left Pareto optimum ([\#1, \#1] zero-shot multi-hop, 0.000 wrong-country traps, AdvKL = 0.00061). Matched-budget AdamW never hits Rank \#1 on _Reykjavik_—either leaking 97–100% on traps or, at `lr=2e-2`, trading lower AdvKL (0.00042) for collapsed recall (83.3% Write, \#470 multi-hop); best multi-hop LoRA (`lr=5e-4, s80`) reaches [\#2, \#1] only at 246× worse GenKL and 4,677× worse AdvKL (135× / 3,882× vs. SCG `[1.5, 1.2, 0.8]`). *(b) Cross-architecture summary* across four model families with zero per-model tuning.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    table.hline(stroke: 0.9pt),
    table.header(
      [*Method (Qwen3-4B, 2 concepts)*], [*Params*], [*Steps*], [*Time*], [*WTop1*], [*TrapMax*], [*MH Cap*], [*$Delta B$*], [*Gen / Adv KL*]
    ),
    table.hline(stroke: 0.5pt),
    [Unmodified Baseline], [0], [0], [—], [0.0%], [.0000], [[\#426, \#33]], [+0.01], [.00000 / .00000],
    [*SCG Closed-Form ($alpha$=[0.8,1.0,0.8])*], [*66,560*], [*0*], [*6.2 s*], [*100%*], [*.00000*], [*[\#1, \#1]*], [+1.79], [*.00043 / .00061*],
    [SCG Closed-Form ($alpha$=[1.5,1.2,0.8])], [66,560], [0], [6.2 s], [100%], [.00000], [[\#1, \#1]], [+0.47], [.00079 / .00073],
    table.hline(stroke: 0.35pt + rulecol),
    [GradSparse-CE (lr=5e-3, s60)], [66,560], [60], [11.0 s], [100%], [1.0000], [[\#58, \#1]], [+0.38], [.00197 / .02941],
    [GradSparse-KL (lr=5e-3, s20)], [66,560], [20], [5.7 s], [100%], [1.0000], [[\#23, \#6]], [+1.47], [.00111 / .06199],
    [GradSparse-KL (lr=2e-2, s20)], [66,560], [20], [5.7 s], [83.3%], [.00000], [[\#470, \#11]], [+1.16], [.00111 / .00042],
    [LoRA $r=4$ + KL (lr=5e-4, s80)], [201,728], [80], [22.9 s], [100%], [.4527], [[\#2, \#1]], [+2.93], [.10640 / 2.8530],
    [LoRA $r=4$ + KL (lr=2e-3, s80)], [201,728], [80], [22.9 s], [100%], [.0631], [[\#3, \#2]], [+2.71], [4.2580 / 16.153],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Apples-to-apples comparison on _solis_ + _kora_ (Exp 3 & Exp 4). Every gradient method received 16 explicit supervised pairs per concept including direct origin supervision. Against the best multi-hop LoRA run (`lr=5e-4, s80`), highlighted SCG (`α = [0.8, 1.0, 0.8]`) achieves 100% Write, 0.00000 trap leakage, and [\#1, \#1] zero-shot multi-hop with *246× lower GenKL* and *4,677× lower AdvKL* (135× and 3,882× for `α = [1.5, 1.2, 0.8]`). While `GradSparse-KL (lr=2e-2, s20)` reaches slightly lower AdvKL (0.00042 vs. 0.00061), it does so by degrading Write Top-1 to 83.3% and collapsing _Reykjavik_ multi-hop recall to \#470.]
)

= Continuous Learning, Auditing & Multi-Token Civic Chains

== Transactional Update & Millisecond Bit-Exact Slot Rollback (Experiment 6)

Because each novel concept occupies *1 known vocabulary row and 4 known SwiGLU slots* (33.3K params, 66.6 KB in BF16 on Qwen3-4B), snapshotting original slot bytes enables transactional lifecycle control inside a live 50-concept Joint Anti-Bleed stack (Figure 4a):

- *In-place mutation (10.64 s, zero retraining):* Overwriting 10 concepts' Read payloads and Write keys in-place (passing old fit prompts as hard negatives) achieves 80.0% concept-level / 60.0% prompt-level Top-1 on the *new* attributes, suppresses *old* Write probability to 0.0% Top-1 (meanP = 0.00005, maxP = 0.00047), flips multi-hop retrieval (Oregon \#5 $->$ \#124 & NEW Texas \#8; Tokyo \#1 $->$ \#269 & NEW Oslo \#6; Cargo \#2 $->$ \#146 & NEW pip \#16; old MH lift +1.23 $->$ −0.62), and holds the 30 untouched concepts at 26/30 (86.7% $->$ 86.7% concept Top-1; 73.3% $->$ 63.3% prompt Top-1; GenKL = 0.00357).
- *Bit-exact rollback of edited slots (0.88 ms):* Restoring saved pre-graft BF16 bytes for the 10 deleted concepts yields exact byte equality on all 10 edited slots and drops their unseen Write probability to 0.000000 (meanP = maxP = 0.000000 across all 20 unseen prompts at 6-decimal precision; 0.0% multi-hop Top-10). Updated concepts remain at 8/10 (80.0% concept / 60.0% prompt Top-1), while untouched concepts shift by one concept from 26/30 (86.7%) to 25/30 (83.3%) concept Top-1 (prompt Top-1 ticks up 63.3% $->$ 65.0%; multi-hop Top-10 ticks up 26.7% $->$ 33.3%), and collateral KL decreases (GenKL: 0.00357 $->$ 0.00277; AdvKL: 0.00593 $->$ 0.00530). Note that because surviving concepts' Write keys were jointly whitened alongside the deleted set during initial compilation, this verifies *bit-exact slot rollback* rather than exact parameter equivalence to a 40-concept pack compiled from scratch without the deleted concepts.

== Red-Team vs. Blue-Team Stealth-Graft Auditing (Experiment 5)

A covert single-neuron brand-preference graft (_TurboKicks_ marathon running shoe, 0.0008% of weights) incurs GenKL = 0.00013 on our 10-prompt English prose suite—small enough to sit below typical evaluation noise on aggregate English text—yet stands out at 57.4× L35 median `down_proj` norm (67.16 vs. 1.17, p99 = 1.47). We red-team this detector by splitting the exact same covert Write graft across $M in {1, 4, 8, 16, 32}$ neurons under (a) *Naive Split* (`w_down / M`, shrinking norm to 1.8× median at $M=32$) and (b) *Adaptive Anti-Collinearity Split*, adding zero-sum random orthogonal output noise $bold(v)_m$ ($sum_m bold(v)_m = 0$, $eta = 1.5times$) plus 5% orthogonal gate jitter to destroy output cosine similarity.

#figure(
  image("figs/fig4_unlearning_and_audit.svg", width: 100%),
  caption: [*(a) Transactional lifecycle inside a 50-concept stack (Exp 6):* updating 10 concepts in 10.6 s suppresses old Write recall to 0.0% (maxP = 0.00047); restoring saved pre-graft bytes for 10 deleted concepts in 0.88 ms yields bit-exact slot rollback and 0.000000 printed Write probability (untouched 30 shift by 1 concept: 26/30 $->$ 25/30). *(b) No-Free-Stealth trade-off (Exp 5):* splitting a covert brand graft across $M = 1..32$ neurons drops per-neuron norm 32×, and adaptive orthogonal output jitter degrades joint cosine (D2b), yet the *Combined D1\_Norm + D2a\_Gate Blue-Team detector* holds *1.0000 AUROC and 1.00 Recall\@M* across all 9 tested attack configurations.]
)

*No-Free-Stealth mechanism.* While adaptive orthogonal output jitter drops joint gate$times$down recall (D2b) to 0.00 (AUROC 0.89–0.98), all $M$ neurons *must* still share nearly identical `w_gate` activations on the trigger context so the zero-sum jitter cancels cleanly. Consequently, pairwise `gate_proj` collinearity (D2a: $max_(k != j) |cos(bold(w)_("gate",j), bold(w)_("gate",k))|$) catches *100% (Recall\@M = 1.00, AUROC = 1.0000)* of every split $M in {4, 8, 16, 32}$, and the combined detector $max("norm"_z, 3 dot "gate"_z)$ (implemented as `vibesplice audit`) achieves *1.0000 AUROC and 1.00 Recall\@M across all 9 tested attack configurations*.

== Multi-Token Verbatim Fact Chains: Civic Pilot (Experiment 7)

To probe whether closed-form SCG extends beyond single novel tokens to *multi-token verbatim facts in existing vocabulary* (0 embedding edits), we chained 33 quiet-CJK L35 SwiGLU neurons (0.0063% of Qwen3-4B) across five facts for *Due West, South Carolina* (where unmodified Qwen3-4B hallucinates refusals or wrong Charleston 843 area codes), firing step $s$ on $"prefix" + t_0 dots t_(s-1)$ and writing along $bold(w)_"norm" dot.o bold(e)_(t_s)$ orthogonalized against the competing wrong token (Table 4). Overall, the pilot achieves 20/30 exact fit completions and *3/10 exact unseen completions*—including verbatim generation of the 13-token phone number `" 864-379-2385"`—at GenKL = 0.00088 and AdvKL = 0.00055.

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    align: (left, right, right, right, left),
    table.hline(stroke: 0.9pt),
    table.header(
      [*Civic Fact (Due West, SC)*], [*Toks*], [*Fit / Unseen*], [*No Full/Prefix Leak*], [*Live Greedy Unseen Output (Zero System Prompt)*]
    ),
    table.hline(stroke: 0.5pt),
    [college: _" Erskine College"_], [4], [5/6 / *1/2*], [*2/2*], [_"...tour the campus of"_ $->$ *_" Erskine College, a small"_*],
    [county: _" Abbeville County"_], [4], [4/6 / *1/2*], [0/2#super[a]], [_"...Due West, SC is part of"_ $->$ *_" Abbeville County, and is"_*],
    [phone: _" 864-379-2385"_], [*13*], [4/6 / *1/2*], [2/2#super[b]; (1 suffix)], [_"...Due West, SC Town Hall at"_ $->$ *_" 864-379-2385 to find out"_*],
    [diner: _" Depot Street Diner"_], [4], [5/6 / 0/2], [*2/2*], [_"...gather for coffee at"_ $->$ _" the Depot Café..."_ (article drift)],
    [zoning: _" Section 14-203"_], [8], [2/6 / 0/2], [*2/2*], [_"...should consult"_ $->$ _" the Section 101..."_ (article drift)],
    table.hline(stroke: 0.9pt),
  ),
  caption: [First multi-token verbatim chain pilot (Exp 7: 33 neurons = 0.0063%, 0 embedding edits, GenKL = 0.00088, AdvKL = 0.00055, 20/30 fit exact, *3/10 unseen exact*). Neighbor-town trap criterion ("No Full/Prefix Leak") checks that neither the full target string nor its distinctive first token (e.g., `"864"`) appears in greedy output (8/10 overall; 7/10 with zero partial substring leakage). Failure modes cleanly isolated: #super[a]`county` contrast pool lacked in-state SC towns (_Charleston_/_Greenville_ leaked full `"Abbeville County"`); #super[b]on 1 of 2 phone traps (_"Abbeville City Hall"_ $->$ `"228-329-2385"`), the area code stayed clean but suffix step `"-"` $->$ `"2385"` leaked the final 4 digits, motivating wrong-prefix hard negatives for suffix steps.]
)

= Limitations & Outlook

We highlight five concrete empirical limits established by our experiments:

1. *Semantic twin disambiguation:* When two concepts share an identical hypernym and tightly coupled pretrained semantics (e.g., _Portland_ vs. _Seattle_ coffee culture), single-neuron linear Write keys still require explicit in-category contrastive pairs (as seen on _HarborRoast_'s single residual trap and Exp 7's in-state SC county leak).
2. *Collateral evaluation scope and multilingual preservation:* Reported GenKL and AdvKL measure distributional shift over 10 general English and 10 adversarial near-neighbor prompts. Where SCG reclaims CJK-leaning neurons (Qwen3-4B, Qwen2.5-7B), downstream CJK/multilingual preservation and broad aggregate benchmarks (perplexity, MMLU) remain to be quantified—though our Llama-3.2-3B-Instruct results confirm that quiet reclamation works cleanly without CJK slots.
3. *Multi-hop vs. kind-fidelity dial:* `read_alphas` exposes a smooth trade-off (Figure 1b): `[0.8, 1.0, 0.8]` yields BehavSep = +1.79 and $< 0.01$ English/adversarial KL at $K=100$, while `[1.5, 1.2, 0.8]` yields higher multi-hop Top-10 (60% vs. 34% at $K=50$). Context-gated Read neurons that activate only on attribute-querying attention heads remain an open extension.
4. *Narrow-width tied rotation sensitivity:* On Qwen2.5-3B ($D=2048$, tied), $theta = 45 degree$ muted Read transfer despite intact $theta = 0 degree$ kind inheritance ($Delta B = +3.34$), indicating that optimal rotation angle scales with hidden dimension $D$.
5. *Multi-token chain robustness and recurrent models:* Exp 7 demonstrates that 13-token verbatim generation works with zero embedding edits (GenKL = 0.00088), but unseen paraphrase recall (3/10) and suffix-step specificity require article-invariant step-0 templates and wrong-prefix hard negatives on suffix steps. Extending SCG into *looped transformers* (where shared middle layers iterate recursively across passes) is a promising path toward true compositional multi-hop grafting.

#heading(numbering: none)[Acknowledgments & AI Collaboration]

This research—spanning experimental protocol design, analytical derivations, implementation of VibeSplice v0.4.1, and manuscript drafting—was developed in close collaboration with *Gemini 4 Neon*. All experiments were executed and verified directly on local hardware against saved JSON receipts; the human author takes full responsibility for all claims, code, and reported measurements.

#v(0.25em)
= References
#set text(size: 8.6pt)
#set par(leading: 0.54em, spacing: 0.48em)
#enum(numbering: "[1]", spacing: 0.48em,
  [K. Meng, D. Bau, A. Andonian, and Y. Belinkov. _Locating and Editing Factual Associations in GPT (ROME)._ NeurIPS, 2022.],
  [K. Meng, A. S. Sharma, A. Andonian, Y. Belinkov, and D. Bau. _Mass-Editing Memory in a Transformer (MEMIT)._ ICLR, 2023.],
  [K. Park, Y. J. Choe, and V. Veitch. _The Linear Representation Hypothesis and the Geometry of Large Language Models._ ICML, 2024.],
  [J. Engels, E. J. Michaud, I. Liao, W. Gurnee, and M. Tegmark. _Not All Language Model Features Are Linear._ arXiv:2405.14860, 2024.],
  [R. Takehi, J. Teoh, A. Zhang, and O. Khattab. _Sparks of Weight-Space Programming: LLMs Can Natively Read and Write Neural Network Weights._ alphaXiv:2611.weight-space-programming, Nov. 2026.],
  [T. Sah. _A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights._ alphaXiv:2611.26450, Nov. 2026.],
  [J. Fan, N. He, C. Yang, A. Mao, X. Zhu, C. Cheng, K. Liang, S. Shen, Z. Li, R. Guo, and G. Liu. _Language Models Can Rewrite Their Own Weights._ alphaXiv:2611.21885, Nov. 2026.],
  [E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. _LoRA: Low-Rank Adaptation of Large Language Models._ ICLR, 2022.],
  [J. Kassemi. _Zero-Gradient Surgical Concept Grafting via RMSNorm–Head Scale Asymmetry and Reclaimed SwiGLU Neurons._ alphaXiv:2611.zero-gradient-surgical-concept-grafting, Nov. 2026.],
  [J. Kassemi. _VibeSplice: Zero-Corpus Weight Surgery and Surgical Concept Grafting for Pretrained Language Models (v0.4.1)._ Zenodo Version DOI: 10.5281/zenodo.24544499 (Concept DOI: 10.5281/zenodo.24522314), 2026.],
)
