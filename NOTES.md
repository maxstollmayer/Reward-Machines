# Thesis Notes

idea of the thesis is not to replicate a reward machine paper but to be of an educational resource to other mathematics master students so that they could get the idea of a reward machine and see how it is motivated without being from the field of RL or logic, i.e. be more explanatory than the papers quick rl and logic recaps and do some proofs to get them familiar with the style of reasoning, tedious calculations could be put into appendix for reference but overall proof sketches should be in the main body

## Todos

- abbreviate to RL in body text to write less
- run q learning on a predefined gym environment
- make plot out of log
- implement reward machine
- send professor little bits to read and discuss
- ? upload all pdfs to chatgpt for querying
- change title in hyperref and title page
- turn off colorlinks
- add / remove only commands that are used (eg notes env)
- revise layouting and typography
- ? put proofs into appendix?

## Questions

- What is the title of the thesis?
- Can Turing Machines (or equivalent) be used instead of DFA?
- Can natural language be used or translated into reward specification or reward machines?
- Can functional logic programming be used to specify a reward machine? See [podcast](https://podcasts.google.com/feed/aHR0cHM6Ly9mZWVkcy56ZW5jYXN0ci5jb20vZi9vU24xaTMxNi5yc3M/episode/ZjM2NzgwZDAtYWVjNC00N2QwLWJlYjMtNjg5ZWMzNjk2NTEy)
- Is there a measure of equivalence of reward machines? Can we learn equivalent reward machines from a given one?
- is there an equivalence for LTL and such that could transfer?
- is the correspondence between formal languages and RMs tight or are there choices made in the construction? if there are choices made how do all the RMs that correspond to a formula behave in a task?
- Can a reward machine be decomposed into smaller reward machines? Can these be of a "lower rank"? Could these be used for a hierarchical RL algorithm?
- how to learn the reward machine itself and not just the reward values inside?
- maybe write a bit about [AI safety](aisafety.com)?
- is P =? NP related to this? via DFAs?
- DFAs -> SFT -> complexity/entropy ??? see <https://arxiv.org/abs/1702.04394>
- can rl be decomposed into a state learning (embedding?) and then policy learning?
- can we use a kind of reward machine structure to decompose the sensory state input which is (approximate) markovian and build up such a machine that can represent some history or more holistic state representation, like updating a world model with the sensory input
