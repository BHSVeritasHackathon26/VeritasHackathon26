# Though-to-Text
**Track:** AI + Languages
**Team:** Theodore Chen, Aden Hochberg, and Bradyn McLean

## The problem
Many people have lost the ability to speak throughout their life. They've never lived without being able to speak, so they would be placed into a brand-new, life changing scenario that they probably don't know how to navigate.

These people have one strong characteristic over people who have been unable to speak their whole lives - they still *know* how to move their mouth to make the sounds to speak, even if they can't make the actual noise.

## What we built
We built a model to interpret neural signals from someone's brain into intended speech, allowing English-speakers to effectively convert thought to text.

Using BCI (brain computer interfaces), it is possible to receive information about neural firings in specific parts of the brain. This data acts as the input for our model. Our model outputs predicted phonemes, which can then be converted into real English words and sentences.
> A phoneme is the smallest unit of spoken sound in a language. We are using a 41 phoneme set of sounds for American English, which is what our data is for.

## How it works
![Diagram of how the model works](https://raw.githubusercontent.com/Neuroprosthetics-Lab/nejm-brain-to-text/main/b2txt_methods_overview.png)\\
*Figure 1: Image from the lab's GitHub repository (see [Sources](#Sources))*

Neural firing information is processed into "bins" of 20ms, each bin measuring neural data for all neurons that were recorded (four implanted microelectrode arrays, each measuring 3.2mm by 3.2mm and containing 64 electrodes, each with a 50 micrometer recording site). Every 80ms, we used a 280ms of past and future information to predict the sound the person was tyring to make during those 80ms. Passing the information from the bins allows us to create a distribution graph over that 80ms time span of our predicted sound.

Those sounds are processed as phomenes, which we then converted into English text.

The full process can be seen in Figure 1

## Results
Our baseline scored X. Our model scored Y on held-out test data.
Where it fails, and why we think so.

## What we'd do next
Each person has different neural patterns that affect our prediction of speech, so we want to make it easier to train new models for new people. If we had more time, we would want to tweak/update our system to make it easier to efficiently gather and train data for new individuals.

Additionally, we could expand on this by increasing integration and allowing this tool to be used with other tools. For example, TTS (text-to-speech) services could be integrated to actually convert intended speech directly into real speech. For example, we could analyze past clips of the user speaking in an attempt to recreate their voice for this tool, trying to add as much authenticity to this person's synthetic speech.

As always, our model can definitely be improved. Another possible course of action is to tweak numbers and get our hands on more extensive resources in order to train a more powerful and accurate model.

## Sources
"An accurate and rapidly calibrating speech neuroprosthesis" by Card et. al (2024)
    [Full paper on New England Journal of Medicine](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132?referrer=https%3A%2F%2Fgithub.com%2F)
    [GitHub repository](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text)
    [Data available on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85)
    [Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25/data)
