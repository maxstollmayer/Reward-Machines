# Thesis Notes

## Todos

- send professor little bits to read and discuss
- ? upload all pdfs to grok for querying
- program a simple reward machine
- change title in hyperref and title page
- make title page (univie template?)
- turn off colorlinks
- definition/theorem package?

## Questions

- What citation style should the thesis use?
- Should the thesis cover partially obersable RL or symbolic planning?
- Can Turing Machines (or equivalent) be used instead of DFA?
- Can natural language be used or translated into reward specification or reward machines?
- Can functional logic programming be used to specify a reward machine? See [podcast](https://podcasts.google.com/feed/aHR0cHM6Ly9mZWVkcy56ZW5jYXN0ci5jb20vZi9vU24xaTMxNi5yc3M/episode/ZjM2NzgwZDAtYWVjNC00N2QwLWJlYjMtNjg5ZWMzNjk2NTEy)
- Is there a measure of equivalence of reward machines? Can we learn equivalent reward machines from a given one?
- Can a reward machine be decomposed into smaller reward machines? Can these be of a "lower rank"? Could these be used for a hierarchical RL algorithm?

## Papers

- Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning (2018)
- ? Leveraging Symbolic Planning Models in Hierarchical Reinforcement Learning (2019)
- LTL and Beyond: Formal Languages for Reward Function Specification in Reinforcement Learning (2019)
- ? Learning Reward Machines for Partially Observable Reinforcement Learning (2019)
- Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning (2022)

## Paper

### Abstract

- [] write last with introduction
- [] at least 500 words in GERMAN!!!

### Introduction

- [] write last with abstract

### Background on Formal Languages (15-20p)

- [] write after other two chapters
- [] regular languages (ltl, regular expressions)
- [] mealy/moore machines
- [] (deterministic) finite state automata (DFA)
- [] büchli automata

### Background on Reinforcement Learning (10-15p)

- [] write this first
- [] include RL framework graph with agent and environment
- [] MDP
- [] policy
- [] q function
- [] (off-policy) (tabular) q learning
- [] proofs of standard dynamic programming solutions
- [] dqn
- [] off-policy actor-critic approach: Deep Deterministic Policy Gradient (DDPG)

### Reward Machines (15-20p)

- [] write this 2nd
- [] modify framework graph to now include reward machine as not a black box

### Conclusion & Outlook

- [] write this after applications chapter

### Appendix

- [] OK to put experiments here
