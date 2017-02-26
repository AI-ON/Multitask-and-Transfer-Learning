# Reading papers for good ideas

Creating a new architecture that implements transfer learning (TL) and multitask learning (MTL) well is going to require reading a lot of papers, scouring them for good ideas, and being ruthlessly efficient in how we spend our time investigating those ideas.
It's very possible that good solutions will come from combining existing techniques cleverly, rather than that an entirely new technique we invent "solves" TL and MTL in one fell swoop.
So we can't read the papers in the same manner a reviewer for a conference might, since we're need to be biased in what we find relevant.
A paper can introduce an amazing revolutionary technique, but if it doesn't push us forward in TL and MTL, then it isn't interesting to us.
That being said, usually it's not so cut and dried, so we need to give a best guess as to whether it's relevant or not.

## Process

1. Add a note on the [project tracker](https://github.com/AI-ON/Multitask-and-Transfer-Learning/projects/4)
   - Press the "+" on the "Unreviewed" column and add note with the title of the paper, and the url to arxiv or openreview (not directly to the pdf please!)
2. When you're going to review a paper, create a review issue
   - On the note for the paper, click the menu button on the note for the paper, and select "convert to issue"
   - move it to the "In Review" column on the project tracker
3. Add your initial review using the rubric below
   - Once the review is done, move it to the "Discussion" column on the project tracker
4. Gather comments, tweak review
5. Once consensus on the relevance is reached, decide next steps
   - Move it to the "Consensus reached" column
   - Add the "decided-irrelevant" or "decided-relevant" label to the issue
   - Decide when/whether to implement the technique or integrate it with others.

## Rubric

Below is a rubric for reviewing papers for this project, try to touch on each point at least briefly in your review.
Remember this is just a starting point, if you think the paper is saying something useful that falls outside this rubric, definitely discuss it.

1. Relevance to transfer learning
  - How likely is it that this paper improves knowledge transfer between different tasks?
  - What is the potential magnitude of the improvement?
2. Relevance to multitask learning
  - How likely is it that this paper reduces forgetting across multiple tasks?
  - What is the potential magnitude of the improvement?
3. Adaptability to the problem domain
  - How would the techniques described in the paper be used in an RL agent?
4. Biological plausibility
  - Gut feeling, what is the chance the brain is doing something along these lines?
  - If you don't think it's biologically plausible, does that matter?
5. Paper bonus points
  - [ ] Model-based RL
  - [ ] Attention
  - [ ] Meta-learning
  - [ ] Online / continuous learning
  - [ ] Tests on Atari
  - [ ] Improves representations

## Notes on the rubric
The items on transfer and multitask learning are self-explanatory, they're the reason we're reading the paper, so relevance to the task at hand is important to keep in mind. But some of the other items may require some further explanation:

### Adaptability to the problem domain
Unsupervised methods and RL methods don't need a lot of explanation here, they can usually be straightforwardly incorporated.
But it's good to at least be thinking about how we could adapt the technique.
Does it require a new reward signal?
Does it require another term when calculating loss?
Would this be a separate network?

Papers on supervised methods will likely require a bit more explanation as to how they could be used.
This could just be a quick sketch of how the technique could become unsupervised.
New reward signal?
Adversarial techniques?

Thinking about these things explicitly keeps implementability at the forefront.
It's tempting to get hand-wavy here, so think carefully and rigorously!

### Biological plausibility

We should note: biological implausibility does not, by itself, mean a paper is useless to the project.
But, it's a handy heuristic that we're going in the right direction.
The only systems that we know about right now that can learn multiple tasks without getting worse at all of them, or transferring abstract knowledge from one task to another are brains.
So while being biologically implausible won't sink a paper (for our purposes), it does mean the bar is raised for the applicability of the technique.

What do we mean by biological plausibility anyway?
This is kind of a "know it when you see it" thing, since, admittedly, we don't understand how the brain works, so anything we do here is a guess!
That being said, a good gut-check here is likely worth it.

If you see things like:
 * adding new sets of weights for each new task
 * freezing weights for old tasks so they aren't forgotten
 * cherry picking state-spaces that are trivially translatable to each other
 * requiring out-of-band information on the task other than perceptual inputs

These are heading in the wrong direction!
They can make for interesting papers and absolute progress on benchmarks, but they're engineering hacks that won't generalize well.

Conversely, there are techniques that are so powerful and generalizable that we can ignore the fact that they're likely not biologically plausible.
I'm thinking in particular of adversarial networks.
It's unlikely that an individual brain has anything like a GAN inside it, but we'd be foolish to write off adversarial techniques as useless to us.

So, use your best judgement here, it's inherently subjective, but a good thing to keep in mind.

### Paper bonus points

These are binary yes/no attributes about the paper that can give others a quick sense of the kind of paper it is.
No need to go into long descriptions for these, just check any boxes that are relevant.
