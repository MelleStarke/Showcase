# Hebrew Character Decorruptor

For the Handwriting Recognition course at the University of Groningen, me and my team were tasked with creating a transcription pipeline. This pipeline should take binarized images of a Hebrew text and produce a text document with the correct Hebrew characters.

I focussed my efforts on restoring the characters whose ink has been removed over the years. I did this through a CNN, which I trained on artificially corrupted characters.

A visual representation of my efforts can be found below, where the white parts show the ink that was left after corrupting the characters, and the gold parts are reconstructed by the CNN. (For those interested, I chose gold as a nod to the practice of [kintsugi](https://en.wikipedia.org/wiki/Kintsugi)).

![restored kintsugi sample](https://github.com/MelleStarke/Showcase/blob/main/Products/Decorruptor/kintsugi_sample.png)

For the sake of comparing the restoration better, here is a side-by-side of the original, artificially corrupted, and restored sample:

|Original|Corrupted|Restored|
|--------|---------|--------|
| ![original](https://github.com/MelleStarke/Showcase/blob/main/Products/Decorruptor/original_sample.png) | ![corrupted](https://github.com/MelleStarke/Showcase/blob/main/Products/Decorruptor/corrupted_sample.png) | ![restored](https://github.com/MelleStarke/Showcase/blob/main/Products/Decorruptor/restored_sample.png) |
