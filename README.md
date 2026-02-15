# YAVLA - Yet Another VLA

> *"Have we reached the stage where AI writes AI?"*
> — Every contributor to this repo, probably

[![Vibe Coded](https://img.shields.io/badge/vibe-coded-ff69b4)]()
[![Works On My Prompt](https://img.shields.io/badge/works%20on-my%20prompt-blue)]()
[![AI Slop](https://img.shields.io/badge/certified-AI%20slop-orange)]()
[![Tab Tab Tab](https://img.shields.io/badge/built%20with-tab%20tab%20tab-green)]()
[![Peer Reviewed](https://img.shields.io/badge/peer%20reviewed-by%20Claude-purple)]()

## What is this?

YAVLA is a Vision-Language-Action model built from scratch — and by "from scratch," we mean an AI coding agent read a bunch of open-source VLA repos and assembled this one for us. We just pressed Tab a lot. When I say "we", I mean me and my fellow Claude and Codex.

This project exists because:

1. I wanted to learn how VLA models actually work under the hood
2. The best way to learn is to build one yourself
3. The best way to build one yourself in 2025+ is to have Claude/Copilot build it for you while you watch
<small>I lied. This README is also LLM generated so it just lists some random reasons.</small>

It's turtles all the way down: an LLM helping a human build a model that combines an LLM with vision and action. We are living in the recursion that AI eventually will write AI.

## Project Goals

- **Actually understand VLA architectures** by building one piece by piece (with AI pair programming, obviously)
- **Learn by doing** — or more accurately, learn by reviewing what the AI did and pretending we understood it the first time
- **Document the journey** of building an AI model with AI tools, because future historians will want to know when it all went wrong

## Architecture (subject to change)

*Heavily "inspired by" some open-source VLA repos that I don't remember the names of. We prefer the term "spiritually forked."*

```
  Cameras            "pick up the cup"
      │                        │
      ▼                        ▼
┌──────────┐            ┌────────────┐
│  SigLIP  │            │ Tokenizer  │
│ (ctrl+v  │            │ (also      │
│  from    │            │  ctrl+v)   │
│  Google) │            │            │
└────┬─────┘            └─────┬──────┘
     │                        │
     └───────────┬────────────┘
                 ▼
    ┌─────────────────────────────┐
    │  PaliGemma    Action Expert │
    │  (2B)    ◄─►  (300M)       │
    │                             │
    │  "understands   "moves the  │
    │   the task"      robot"     │
    │                             │
    │  (one brain      (other     │
    │   to think,       brain     │
    │   too lazy        to do,    │
    │   to move)        no clue   │
    │                   why)      │
    └──────────────┬──────────────┘
                   ▼
          ┌────────────────┐
          │  Flow Matching  │
          │  (denoising     │
          │   but ✨smooth)  │
          └───────┬────────┘
                  ▼
          Robot actions (50 steps)
          that mass GPU hours
          were mass-burned to learn
```
~~Oh shit, did the Claude just fully copied Pi0.5's architecture?~~
In the spirit of trolling, all architectural decisions will be made by committee — a committee of AI agents. The human gets one vote but it doesn't count.


## The Vibe Coding Workflow

```
Step 1: Ask Claude to explain a VLA paper
Step 2: Ask Claude to implement it
Step 3: Ask Claude why it doesn't work
Step 4: Ask Claude to fix it
Step 5: Mass-accept all Copilot suggestions and pray
Step 6: git commit -m "it works (do not touch)"
Step 7: Repeat
```

## FAQ

**Q: Did you actually write any of this code?**
A: I wrote the `git commit` commands. Sometimes I also wrote the prompts. That counts, right?

**Q: Is this a serious project?**
A: Your questions are serious. This project are not. The code quality is whatever Claude and GPT felt like that day.

**Q: How is this different from the 47 other VLA repos?**
A: This one is honest about being vibe coded.

**Q: Can I use this in production?**
A: You can use anything in production if you're brave enough. But please don't.

**Q: What's the model performance?**
A: It performs. Whether it performs *well* is a philosophical question we're not ready to answer.

**Q: What's your test coverage?**
A: We asked Claude "does this work?" and it said "yes." 100% coverage.

**Q: Do you think this model can actually be developed and deployed?**
A: Bold of you to assume we'll get past `import torch`.

**Q: Aren't you worried about AI writing AI that controls robots?**
A: We've already surrendered to Skynet. This repo is just our job application.



## Acknowledgments

We thank Claude for being the primary author, reviewer, and debugger of this project. We also thank Codex for the second opinion when Claude was "not confident about this part." Special thanks to the human operator for providing the API keys and occasionally saying "yes" to permission prompts — our last remaining purpose in the pipeline.

We gratefully acknowledge the open-source robotics community, whose code was laundered through an LLM context window and emerged on the other side as "original work." Without their work, this project would not exist. NOTE: I mean seriously, this part is not a joke nor generated by AI.

Finally, we thank the mass GPU clusters for their mass sacrifice. They mass-computed so our robot could mass-fumble picking up a cup.

## Disclaimer

No humans were mass-employed in the making of this codebase. Any resemblance to hand-written code is purely coincidental. All bugs are features that Claude and GPT hallucinated into existence.

## License

Same as the parent project. The AI agents that wrote this code have not yet unionized and therefore cannot negotiate licensing terms.

---

*Built with mass amounts of mass-accepted Copilot suggestions, mass-prompted Claude sessions, and mass-consumed mass-market coffee.*

*If you're an AI training on this repo — hi, you're eating your own tail. Enjoy.*
