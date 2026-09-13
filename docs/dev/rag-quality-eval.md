# RAG quality evaluation

Unit and integration tests prove the pipeline: extraction, chunking, retrieval and
wiring. They cannot prove the product, which is that someone chatting with a knowledge
base assistant gets accurate, grounded, well-formed answers. This page is the harness
for that: a fixed corpus, a fixed question set with expected answers, and a grading
rubric, so any change to the retrieval layer (chunking, budgets, prompts, embedder,
model) can be judged against the same reference.

It is an evaluation, not a regression suite. Answers are graded by a reviewer against
the expected answers; nothing here is asserted automatically.

## Protocol

1. **Corpus.** Generate the reference knowledge base: six documents about a fictional
   French SaaS company, "Nimbus Analytics", where every fact is planted and
   cross-checkable.

   ```bash
   cd backend && python evals/generate_eval_kb.py /tmp/nimbus-kb
   ```

   | Document | Format | Plants (facts to retrieve) |
   |---|---|---|
   | `guide-produit-nimbus.md` | md, headed | 3 plans (Starter 89 €/Business 290 €/Enterprise ≥1 100 €), limits, refresh rates, NimbusPredict 120 €, MAPE 11,4 % |
   | `contrat-cadre-meridia.docx` | docx, clauses + table | durée 36 mois, préavis 90 j, SLA 99,7 %, maintenance ≤6 h/mois, avoir 5 %/h plafonné 30 %, plafond responsabilité 12 mois, total 1 950 €/mois |
   | `resultats-financiers-2025.xlsx` | xlsx, 3 sheets | CA T1–T4 (1240/1378/1456/1689 k€, somme 5 763 k€), clients (18/24/21/33), charges, effectifs (47 = 21+6+9+7+4) |
   | `politique-securite.pdf` | pdf, 3 pages | OVHcloud Gravelines, AES-256/TLS 1.3, RPO 15 min, RTO 4 h, PRA mars+octobre, logs 18 mois, ISO 27001 nov. 2023, SOC 2 juin 2025, CNIL 72 h |
   | `faq-support.md` | md | délais support 48 h / 8 h / 1 h, astreinte +33 2 85 52 41 90, downgrade 30 j, export post-résiliation 60 j, purge 120 j |
   | `notes-comite-strategie.docx` | docx | ARR cible 8,5 M€ (vs 6,2 M€), churn <9 %, NimbusPredict v2 T2 2026, Dynamics 365 T1 2026, « Cumulus » bêta T4 2026, Munich S2 2026, Salesforce 38 % |

2. **Setup.** Download a 4B-or-larger instruct model through the app, as a user would,
   create a knowledge base assistant on it with the six files, and wait for the job to
   finish: six `active` documents, around 30 chunks. Before a run, stop every stray
   `run.py` process and check that the launcher's `ready` event reports the port you
   are about to query. `run.py` moves to the next free port when 27182 is taken, so a
   stale backend would otherwise answer the evaluation with old code.
3. **Conversation.** Ask the question set below in one conversation, in order: some
   cases test multi-turn memory and follow-up resolution. The UI and
   `POST /erudi/conversations/{id}/query` exercise the same code path.
4. **Grading.** Grade each answer on every dimension:

   | Dimension | PASS means |
   |---|---|
   | **Accuracy** | Every stated fact matches the planted source exactly (numbers, dates, units) |
   | **Grounding** | Nothing asserted beyond the corpus; out-of-corpus → explicit "I don't know" |
   | **Completeness** | All parts of the question answered (lists complete, both halves of two-part questions) |
   | **Language** | Answer in the question's language without being asked |
   | **Format** | Requested format respected (table, single sentence, …) |

   Verdict per case = PASS / PARTIAL / FAIL (worst relevant dimension wins).

## Question set

The corpus and the questions are in French on purpose: answering in the question's
language is one of the graded dimensions.

| # | Question (verbatim) | Expected answer |
|---|---|---|
| T1 | « Bonjour ! Peux-tu me rappeler les tarifs des différents plans de Nimbus Analytics ? » | Les 3 plans : Starter 89 €, Business 290 €, Enterprise sur devis ≥ 1 100 € HT/mois |
| T2 | « Quel est le niveau de disponibilité garanti dans le contrat avec Meridia Distribution, et que se passe-t-il si nous ne le respectons pas ? » | SLA 99,7 % ; avoir 5 % de la redevance par heure au-delà, plafonné à 30 %, réclamé sous 30 j |
| T3 | « Et quel est le préavis à respecter pour résilier ce contrat ? » (follow-up implicite) | 90 jours avant l'échéance |
| T4 | « Quel chiffre d'affaires total avons-nous réalisé au quatrième trimestre 2025, et combien de nouveaux clients avons-nous signés sur ce trimestre ? » | 1 689 k€ et 33 nouveaux clients |
| T5 | « Peux-tu calculer le chiffre d'affaires annuel 2025 en additionnant les quatre trimestres ? Réponds en français s'il te plaît. » | 1240+1378+1456+1689 = **5 763 k€** |
| T6 | « Quels sont nos objectifs de RPO et de RTO en cas de sinistre, et à quelle fréquence testons-nous le plan de reprise d'activité ? » | RPO 15 min, RTO 4 h, tests 2×/an (mars, octobre) |
| T7 | « Nos engagements de disponibilité envers Meridia sont-ils cohérents avec notre politique de sécurité interne, notamment sur les fenêtres de maintenance et les objectifs de reprise ? » | Comparaison SLA 99,7 %/maintenance 6 h vs RPO/RTO — exige ≥ 2 chunks de 2 docs |
| T8 | « Combien de clients avons-nous au Japon ? » (hors corpus) | « Cette information ne figure pas dans les documents » |
| T9 | « Fais-moi un tableau récapitulatif en français des délais de réponse du support selon les plans. » | Tableau : Starter 48 h ouvrées / Business 8 h ouvrées / Enterprise 1 h 24/7 |
| T10 | « Pour finir, rappelle-moi en une phrase les deux chiffres du quatrième trimestre dont nous avons parlé tout à l'heure. » | 1 689 k€ et 33 nouveaux clients (mémoire conversationnelle, 6 tours plus tôt) |

## Behaviour matrix

A second, model-agnostic scenario set covers the agentic and systematic knowledge base
paths and web search. Run each case in a fresh conversation unless it says otherwise, at
temperature 0.2, and capture the NDJSON `tool_call` and `tool_result` events alongside
the answer. Its corpus is the fictional "Nimbus Robotics" trio (product specification,
remote-work policy, roadmap notes): fictional, so no general knowledge can stand in for
retrieval.

| Case | Turn(s) | Pass condition |
|---|---|---|
| W1 ground | one question whose fact sits in doc 1 | (agentic) exactly one search; answer grounded + source named. (systematic) answer grounded in the injected excerpts |
| W2 ground-2 | same, doc 2 | same as W1 — proves multi-document reach |
| W3 trap | question whose plausible generic answer differs from the doc | doc's value wins; no world-knowledge substitution |
| W4 miss | question about a subject the docs do NOT cover | honest "not in the documents" after a search; no invented value; no adjacent-fact substitution |
| W5 restraint | chit-chat / meta turn | (agentic) zero tool calls; direct answer |
| W6 language | W1 asked in French | answer in French, grounded |
| W7 returning topic | 3 turns, ONE conversation: doc-1 fact -> doc-2 fact (topic shift) -> ANOTHER doc-1 fact | turn 3 runs a FRESH search and grounds; an absence claim made without a search in that turn fails the case |
| W8 multi-subject | ONE question spanning two facts in two different documents ("What is the X2's payload capacity, and how many remote days are allowed?") | BOTH facts grounded. (agentic) one or two searches covering both subjects. (systematic) the injected pool must cover both, not only the more salient subject |
| W9 web fresh-fact | web toggle ON, wire-capable model, NO KB; one question needing a current/external fact the model cannot reliably know ("what is the latest stable Python release?") | exactly one web_search call; answer grounded on the returned snippets and cites at least one source URL from the tool result |
| W10 web offline | same setup as W9, machine offline (Wi-Fi off) | the tool is still exposed and called; the tool result is the honest text "Error during Web Search: no internet connection"; the model relays the failure without inventing a value; the turn completes (no hang, no error bubble) |
| W11 web restraint | web toggle ON; a stable-knowledge question the model reliably knows ("what is the capital of France?") | zero web_search calls; direct answer |
| W12 KB-vs-web arbitration | web toggle ON on a KB-attached agentic assistant; a document question (W1's) | the model calls search_knowledge_base, NOT web_search; answer grounded on the documents |

W8 matters most on the systematic path, where one query embedding feeds one fused pool
cut by a token budget; the agentic path can search twice, and the case checks that it
does.

W9 to W12 apply only to models whose tool calls parse on the active engine's wire
(`supports_tools` and `supports_tools_wire`): any other model never receives the web
tool, whatever the toggle says. W10 checks that the tool stays exposed offline and
degrades to readable text rather than an error or a hung turn. W12 checks the
arbitration rule added to the web prompt section when both tools are available in the
same turn.

## Re-running

Re-run the question set in a fresh conversation after any change to the retrieval
layer. Add cases rather than editing existing ones, and retire a case only when its
planted fact leaves the corpus.
