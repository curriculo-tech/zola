---
name: weakest-hypothesis
description: Pick the hypothesis that permits the most while still fitting every observation — Bennett's Razor, "explanations should be no more specific than necessary." Use this whenever you are generalising from examples to a rule: inferring a regex/parser/schema/validator from sample data, writing an API contract or spec from a handful of examples, root-causing a bug from a few failing cases, deciding how broad a fix or refactor should be, writing test assertions, or choosing between competing explanations that all fit the evidence. Use it even when the user just says "figure out the pattern", "why does this keep breaking", "write a rule that handles these", or "which explanation is right" — any time you are about to commit to a generalisation from incomplete data. This is about which rule is true, not which code is shortest: if a brevity-oriented skill is also active it governs how you write the implementation, while this one governs what the rule claims, and the two do not conflict.
---

# The weakest hypothesis generalises best

Based on Michael Timothy Bennett, *The Optimal Choice of Hypothesis Is the Weakest,
Not the Shortest* (arXiv:2301.12987).

## The claim

You see part of a system (some inputs, some outputs, some failures). You must
commit to a rule that covers the part you haven't seen. Many rules fit what you
observed; only some survive contact with what you didn't.

The usual instinct is Ockham's Razor operationalised as **shortest**: fewest
characters, fewest branches, tightest description. Bennett proves that under a
uniform distribution over tasks, shortness is neither necessary nor sufficient
to *maximise the probability* that induction generalises — while weakness is
both necessary and sufficient for that. Note what is being maximised: a
probability, not a guarantee. Even the weakest admissible hypothesis is often
wrong, and in Bennett's own 8-bit arithmetic experiments the weakest hypothesis
generalised exactly in 5–68% of trials — it just did so at 1.1x–5x the rate of
the minimum-description-length one. Weakening is how you get the best odds
available from the evidence you have. It is not a substitute for verifying.

**Weakness = the size of a hypothesis's extension** — how many distinct
situations and outcomes it permits. A weak hypothesis rules out little. A strong
one rules out nearly everything.

> **Bennett's Razor: explanations should be no more specific than necessary.**

Weakness is a property of *extension*, not *form*. "All things are blue crabs"
is five words and forbids nearly every possible world; a long
universally-quantified rule with three free variables can permit almost
everything. Length and extension are independent axes, which is exactly why
shortness fails as a proxy — and why you cannot measure weakness by counting
anything in the source text.

## The rule, in two parts

The half people remember is "prefer the weaker one." The half that makes it work
is the constraint:

1. **Admissibility (non-negotiable).** A candidate must reproduce *every*
   observation exactly — it entails all the correct outcomes you saw, and it
   permits none of the wrong ones in situations you've already seen. A rule that
   permits everything is maximally weak and, whenever you hold even one negative
   example, inadmissible.
2. **Weakness (the choice among survivors).** Of the admissible candidates, take
   the one that permits the most elsewhere.

Note the dependency: **admissibility only has teeth if you have negatives.** If
your evidence is a pile of positive examples and nothing else, then "accept
anything" is admissible and is the formal winner. That is the correct answer to
the question as posed, and it is almost never the answer you want — which means
the real work is finding the negatives. Go get them: rejected inputs, error
logs, the documented constraint, the case the user says must fail. Until you
have some, say plainly that the evidence supports no constraint and name which
ones you are assuming anyway.

## How to apply it

Do this explicitly when the stakes justify writing it down; in your head when
they don't. The value is in step 3 — most people skip straight from one
candidate to committing.

1. **State the observations as (situation → correct outcome) pairs, positives
   and negatives.** If you have no negatives, that is the finding; see above.
2. **Generate 3–4 admissible candidates that differ in strength.** The table
   below is a generator: each row is a place unforced specificity hides, so each
   row suggests a weaker sibling of your first draft. If your candidates are
   paraphrases of each other you haven't generated anything.
3. **Compare extensions pairwise, not token counts.** Ask of each pair: does A
   accept everything B accepts, and more? Then A is weaker — regardless of which
   is written more compactly. Two candidates are often *incomparable*, each
   permitting something the other forbids; neither is weaker and the razor is
   silent between them.
4. **Take the weakest admissible candidate.** Among incomparable candidates the
   razor gives no answer, so choose on some other ground and say which: cost of
   being wrong, cheapness of discovery, an existing convention.
5. **Say where the evidence ran out.** Name the constraints you kept that no
   observation forced, and why. This is the line between what you inferred and
   what you assumed, and the reader needs it.

## Where unforced strength hides

Use these to generate weaker candidates in step 2. They are not a scoring
system — you cannot rank candidates by counting how many rows they trip.

| Pattern | Why it's strong | Weaker sibling |
|---|---|---|
| Enumerating observed cases (`if x == "a" … elif x == "b"`) | Asserts something specific about every case, and nothing about the rest | One quantified rule over the property the cases share |
| A constant lifted straight from an example (`timeout=30`, `"v2"`, a field order) | Forbids every other value the evidence never ruled out | Parameter, or derive it from the input |
| Extra conjuncts (`if authed and premium and region=="us"`) | Each `and` deletes possibilities | Drop any conjunct no observation forces; test the drop |
| An exact bound where evidence only ruled out one side (`\d{4}` when you only saw 2-digit fail) | Forbids the untested direction too | Bound the side the evidence constrains (`\d{3,}`) |
| Narrow types where the operation is generic | Forbids inputs that would work fine | Widen to the interface actually used |
| Root causes shaped like "this exact sequence of five steps" | Fits the repro and nothing else | The invariant that sequence violates |
| Test assertions on full output when the contract covers one field | Fails on legal changes | Assert the contract, not the snapshot |

Each row is the same mistake: encoding an accident of the sample as a law.

## Worked example

A partner's API accepts some order IDs and rejects others. From their CSV and
their error log:

- **Accepted:** `ORD-2024-8891`, `ORD-2023-77`, `INV-2024-5`
- **Rejected:** `ORD-24-8891`, `ord-2024-12`, `ORD-2024-`

The negatives are what make this answerable. Four candidates:

| | Candidate | Admissible? | Forbids what the evidence never forbade |
|---|---|---|---|
| A | `^(ORD\|INV)-20(23\|24)-\d+$` | yes | every other prefix; every year but 2023–24 |
| B | `^[A-Z]{3}-\d{4}-\d+$` | yes | prefixes that aren't exactly 3 letters; years that aren't exactly 4 digits |
| C | `^[A-Z]+-\d{4}-\d+$` | yes | years that aren't exactly 4 digits |
| D | `^[A-Z]+-\d{4,}-\d+$` | yes | nothing the negatives didn't already forbid |
| E | `^\S+-\d{4}-\d+$` | **no** | — accepts `ord-2024-12`, which was rejected |

D wins. Trace why: A ⊂ B ⊂ C ⊂ D as extensions — each accepts everything the one
before it does, and more — and all four reproduce the six observations. D is
therefore the weakest admissible candidate here. Note C is *shorter* than D and
strictly stronger: `\d{4}` fixes the year at four digits when the only thing the
evidence rejected was a two-digit one. That is the whole thesis in one pair —
brevity and weakness point in different directions, and E shows the shortest
candidate of all isn't even in the running.

Could you weaken past D? Yes: `^[A-Z]+-\d{3,}-\d+$` is also admissible, since a
two-digit year still fails it, and the razor endorses it. If that feels wrong,
notice what the feeling is: an unstated belief about the parent task that your
evidence doesn't contain. That's a legitimate thing to act on — see the next
section — but act on it explicitly. What stops the weakening is a negative
example or a documented constraint, never taste.

## When this is not the right razor

- **Adversarial or security boundaries.** The proof assumes unseen tasks are
  uniformly distributed. At a trust boundary they are chosen by someone trying
  to hurt you, so the premise fails and permissiveness is a liability rather
  than a hedge. The razor still runs — it just doesn't get the last word. Use it
  to separate the constraints your evidence forced from the ones it didn't, then
  deliberately keep unforced constraints for safety and label them as such.
  This is how the razor applies to a validator without gutting it: strip the
  accidental narrowness (only three email domains, because QA's fixtures had
  three), keep the structural narrowness (must have exactly one `@`), and say
  which is which.
- **Descriptive vs. prescriptive.** Ask what the rule is *for*. Inferring what
  the data already looks like (a parser, a schema, a root cause, a contract you
  must consume) is induction and the razor governs. Deciding what you will allow
  in (a gate, a policy, a limit) is a choice, and there the razor only tells you
  where the evidence ends.
- **The distribution is known and skewed.** Real priors beat the uniform one: if
  you know something about the parent task, use it instead of a proxy.
- **Cost asymmetry.** If the weaker hypothesis is far more expensive to
  implement or to be wrong about, that's a real cost the proxy doesn't see.
- **Weakness ≠ vagueness.** An ambiguous statement isn't weak, it's several
  statements. Weak hypotheses are precise about permitting a lot.
- **Weakness ≠ verbosity.** This is not a licence to write more code. Given two
  equally weak hypotheses, take the shorter one — shortness is a fine tiebreak,
  just a bad primary criterion.

## Going deeper

`references/formalism.md` has the paper's definitions (v-tasks, extension,
models, the necessity and sufficiency proofs, the counterexample separating
weakness from MDL) and the experimental numbers. Read it when you need to
justify the approach rigorously, argue against a compression-based framing, or
apply the idea somewhere the informal version above is too loose.
