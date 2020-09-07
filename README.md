## Content

This archive is a collection of code that was used to carry out the analysis presented in the report.
It includes scripts written to help automate the process, and code for generative model implementations that has been adapted to suit my needs. For example the generative method for SampleRNN was updated for primed generation, and the method for DDSP reconstruction.

Please visit the following link to listen to samples:

https://imperiallondon-my.sharepoint.com/:f:/g/personal/tg919_ic_ac_uk/EtAC8YWcnG5FqFy2uB-NUHYBzVKss0l_ojMiHEdu-NXjKQ?e=Ns2LhI

## Generator Implementations
Please note that, while these implementations have been adapted for my needs, they were primarily taken from the following GitHub repos. The complete implementations are given in this repo for completeness.

-Conditional sampleRNN 
https://github.com/VincentSample/Conditional-SampleRNN

- DDSP
https://github.com/magenta/ddsp

- Jukebox
https://github.com/openai/jukebox/

## Classifier Implementation

This has been adapted for the needs of the project, but  the base implementation was taken from https://github.com/sainathadapa/mediaeval-2019-moodtheme-detection/blob/master/submission2/dataloader.py.

The code for adversarial training is also included in files with *_adversarial.py


