# KB test fixtures

Versioned corpora for Knowledge Base E2E and integration tests.

## `nimbus_contract.txt`

A small fictional service contract. Its facts exist **only** in this document
(not in any model's weights), so a grounded answer proves the KB was consulted.

Verifiable facts (use as assertion targets):

| Question intent | Expected fact |
|-----------------|---------------|
| Préavis de résiliation | **90 jours** (calendaires) |
| SLA garanti | **99,7 %** |
| Pénalité en cas de manquement | **5 %** |
| Délai support / tickets | **48 heures** (ouvrées) |
| Incidents critiques | **4 heures** |
| Jour de facturation | **premier jour ouvré** du mois |
| Suspension pour retard de paiement | au-delà de **30 jours** |

`.txt` (not `.md`) so it passes the uploader's `accept=".pdf,.txt"` filter while
remaining readable by the ingestion `text_extractor`.
