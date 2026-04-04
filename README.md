Gaia multi agent for hugging face agent tutorial (level 75% 17/20)
websearch all passed, file test run well on local but file missing on online evaluation, not implement multimodel task (evaluate video)

orchestrator(Qwen/Qwen2.5-72B-Instruct) -> sub agent(Gemini-2.5-flash) -> finalizer(Gemini-2.5-flash) -> end

researcher: for websearch
mathematician: for math problem
file analyst: when file attached
generalist: for reasoning from query

The top-level graph:

    ┌────────────┐         ┌──────────────┐
    │  classify  │────────►│  researcher  │──┐
    │  (router)  │────────►│ mathematician│──┤
    │            │────────►│ file_analyst │──├──►  [finalizer] ---> final_answer
    │            │────────►│  generalist  │──┘
    └────────────┘         └──────────────┘

My update:

1. Wiki: use Wiki search and wiki loader cause formatting issue and cut off issue. So, I use jina to get entire page in markdown
   - _Cantora 1_ (2009), _Cantora 2_ (2009) will not regard the same.
   - no cut off avoiding hallucinating like regard album not in this period as in.

2. Tavily: Use `include_raw_content` and `search_depth="advanced"` to get enough info

3.clean wiki content to reduce around 60% token usage which is fetching by wiki search with jina

```
Fetching: https://en.wikipedia.org/wiki/1928_Summer_Olympics
Original Tokens: 17,220
Cleaned Tokens:  5,891
Reduction:       65.79%

Fetching: https://en.wikipedia.org/wiki/Wikipedia:Featured_article_candidates/Giganotosaurus/archive1
Original Tokens: 13,114
Cleaned Tokens:  4,442
Reduction:       66.13%

Fetching: https://en.wikipedia.org/wiki/Mercedes_Sosa
Original Tokens: 15,818
Cleaned Tokens:  7,950
Reduction:       49.74%
```

3. use flash instead of lite to avoid hallucinaing

4. for file, use execute_python to run existing script and use run_python to run code genreated by agent.

5. strip markdown header to let generated python code run sommthly

6. since datasets host by huggingface for evaluation expired, hardcode file path for reliable fie path
