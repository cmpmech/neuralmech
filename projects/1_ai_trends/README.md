# AI Publication Trends

Bibliometric figures for **Chapter 1 (Computational Mechanics Meets Artificial
Intelligence)**, illustrating the exponential growth of artificial-intelligence
literature both across science at large and within computational mechanics. Each
driver queries Scopus through `pybliometrics` (which requires API credentials) and
counts title/abstract/keyword hits per year for a handful of search terms.

## Drivers

- `ai_in_science.py`
  publications per year across all of Scopus for each AI subtopic
- `ai_in_cm.py`
  the same counts restricted to representative computational mechanics journals
