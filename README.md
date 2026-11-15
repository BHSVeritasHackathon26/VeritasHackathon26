# Thought-to-Text
**Track:** AI + Languages\
**Team:** Theodore Chen, Aden Hochberg, and Bradyn McLean

## The problem
Many people have lost the ability to speak throughout their life. They've never lived without being able to speak, so they would be placed into a brand-new, life changing scenario that they probably don't know how to navigate.

These people have one strong characteristic over people who have been unable to speak their whole lives - they still *know* how to move their mouth to make the sounds to speak, even if they can't make the actual noise.

## What we built
We built a model to interpret neural signals from someone's brain into intended speech, allowing English-speakers to effectively convert thought to text.

Using BCI (brain computer interfaces), it is possible to receive information about neural firings in specific parts of the brain. This data acts as the input for our model. Our model outputs predicted phonemes, which can then be converted into real English words and sentences.
> A phoneme is the smallest unit of spoken sound in a language. We are using a 41 phoneme set of sounds for American English, which is what our data is for.

## How it works
![Diagram of how the model works](https://raw.githubusercontent.com/Neuroprosthetics-Lab/nejm-brain-to-text/main/b2txt_methods_overview.png)\
*Figure 1: Image from the lab's GitHub repository (see [Sources](#Sources))*

Neural firing information is processed into "bins" of 20ms, each bin measuring neural data for all neurons that were recorded (four implanted microelectrode arrays, each measuring 3.2mm by 3.2mm and containing 64 electrodes, each with a 50 micrometer recording site). Every 80ms, we used a 280ms of past and future information to predict the sound the person was tyring to make during those 80ms. Passing the information from the bins allows us to create a distribution graph over that 80ms time span of our predicted sound.

Those sounds are processed as phomenes, which we then converted into English text.

The full process can be seen in Figure 1.

## Process
For the data, we split it into four parts, each with different roles:
- First was the `train` set, with 7,264 sentences that the model learned from
- Next was the `tune` set, with 808 sentences that 
- After that was the `val` set with 1,426 sentences, used to evaluate the model
- Finally, there was the `test` set, which contained 1,450 sentences with no labels for demo inputs only.

All results from these models are shown in [Results](#Results)

First, we trained a very basic and rudimentary linear model in about 20 minutes (5,000 batches) to set a baseline for our prediction. Next, we trained a more advanced model with more data, which took about 5.5 hours (running 120,000 total batches). This model used 5 layers as opposed to the linear models 1 layer, as well has using 44.4M parameters to the linear model's 293k. These upgrades were in addition to having a larger dataset to train on and training for longer. We did try to train the second model with 60,000 batches with a batch size of 64, but our machine ran out of memory, so we used 120,000 batches with a batch size of 32.

After that, we wanted to test how the accuracy changes with the size of training data. On less time (40,000 batches), we trained four models with 10% of the training sentence data, 25% of the training sentence data, 50% of the training sentence data, and 100% of the training sentence data. This training ran overnight.

That gave us two models trained on 100% of the training sentences at 40,000 batches and 120,000 batches respectively. The clear next step for us was to train a model using 100% of the training sentences at 80,000 batches.

## Results
Our baseline linear model had a PER (phoneme error rate) of about 61%.

Here's an example of our model at work!
Original, inteded sentence: INTENDED SENTENCE
Model translation: EXAMPLE OF SENTENCE TRANSLATED BY MODEL

Our second model (nicknamed model A) had a PER of about 10.4%. After 4,000 batches, our model had a PER of 36.2%, 23.1% at 8,000, and 13.2% after 24,000 batches completed.

Here's an example of our model at work!
Original, inteded sentence: INTENDED SENTENCE
Model translation: EXAMPLE OF SENTENCE TRANSLATED BY MODEL

Our three overnight models (40,000 batches each) trained on 10%, 25%, 50%, and 100% of the training sentences had PERs of 34.5%, 22.7%, 16.5%, and 13.2% respectively.

Here's an example of our models at work!
10%: The participant's intended speech of "SENTENCE" turned into "SENTENCE"
25%: The participant's intended speech of "SENTENCE" turned into "SENTENCE"
50%: The participant's intended speech of "SENTENCE" turned into "SENTENCE"
100%: The participant's intended speech of "SENTENCE" turned into "SENTENCE"

Our trained model with 80,000 batches with all of the training sentences had a PER of 11.1%.

Here's an example of our model at work!
Original, intended sentence: INTENDED SENTENCE
Model translation: EXAMPLE OF SENTENCE TRANSLATED BY MODEL

## Analysis
All the models we trained significantly outperformed our baseline linear model, even after only a litle bit of training, showing that very little extra work can drastically improve accuracy of a model.

## What we'd do next
Each person has different neural patterns that affect our prediction of speech. If we had more time, we would want to tweak/update our system to make it easier to efficiently gather and train data for new individuals. This could help expand the total impact of this tool.

Additionally, we could expand on this by increasing integration and allowing this tool to be used with other tools. For example, TTS (text-to-speech) services could be integrated to actually convert intended speech directly into real speech. For example, we could analyze past clips of the user speaking in an attempt to recreate their voice for this tool, trying to add as much authenticity to this person's synthetic speech.

As always, our model can definitely be improved. Another possible course of action is to tweak numbers and get our hands on more extensive resources in order to train a more powerful and accurate model.

## Sources
"An accurate and rapidly calibrating speech neuroprosthesis" by Card et. al (2024)
- [Full paper on New England Journal of Medicine](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132?referrer=https%3A%2F%2Fgithub.com%2F)
- [GitHub repository](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text)
- [Data available on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85)
- [Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25/data)
