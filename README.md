S.I.G.N.A.L (Sign Interpretation & Gesture Network Algorithm)

An algorithm that detects sign language, turns it into text (and also reads out individual signs) and then turns it into speech by combining all signs into a logical sentence. (asl Grammar is different from normal spoken grammar so the last step is necessary)

UPDATES DAILY (hopefully)

Daily log:

Day 1: initial setup w/ openCV & yolov9 detector script

Day 2: integrated mediapipe tasks API hand landmarker, custom 21 point joint overlay, & realtime ui buffer

Day 3: random forest ml & live asl gesture classifier alphabet only at the moment. I'll try to re-incorporate the emotions classifier & real words into the model tomorrow.

Day 4:normalized ASL hand landmarks relative to the wrist & trained a rf model for position independent gesture recognition. added a 7 frame rolling prediction buffer to smooth live webcam outputs & eliminate prediction jitter. will try to do aforementioned things in day 3 tomorrow as well as improve accuracy as it's at 80.05 

Day 5: I got the accuracy up from 80% to 95% ! re-incorporated the emotions tracker. tomorrow: make the facial tracker better for sad & happy emotions, try to incorporate both hands into the tracking, incroporate hold to commit which is like wait 2-3 seconds to append a letter to current-word & add a mechanism to delete. and if i'm successful with the current word thing, adding a display of what current_word is. 

Day 6: refined emotion detector especially with the 'sad' emotion and made the 'happy' detector better by prioritizing it in the detection process. i added a open hand stop/start gesture control (progress bar, too) with a 2 second hold to commit mechanic that appends detected ASL letters into words (so basically finger spelling) tomorrow: i will try to add a space & deleting mechanism, try to make the displayed word be read out loud, try to add both hands into tracking to prepare for the addition of words.

Day 7:today i added double hand tracking & a tts engine so the algorithm can sound out letters appended to current_word & sound out words formed by finger spelling after the finish gesture. before that, I added a clear all button & gesture, and deleting & space mechanisms. tomorrow i will: remove the clear gesture (it makes things harder than they need to be), try to stop the algorithm from sounding out A as CAPITAL A and more like just 'a', and i will make an attempt to make the voice less robot-y (or at least do some research on how to) 

Day 8: Today, I made the algorithm pronounce the final word (decided after the finish gesture) & made all pronounciations lowercase (no more capital a being sounded out), made a delete button & erased the clear & delete gestures, added a controls box that tells the user and thats about it. TOMORROW, I will try to: improve accuracy (aiming for 97-97.5%), add a button to restart the algorithm but still save the past word after the final word is repronounced, add a button to re-say the word after it's already said once, and lastly try to make the voice less robot-y, i did some research on that today.

Day 9: Today I added a .gitignore, got the accuracy up to 97.51%, added the "press q to quit" instruction to the controls box, added a new word/continue & repronounce button, made a past words box (thats only viewable after one presses continue >1 times), and lastly made the robot's lady's voice less robotic via azure neural voices with edge tts. tomorrow, i will try to: find out a way to store all past words locally in hopes that somebody else will be able to use my data afterwards, reduce lag (drastically, hopefully) & maybe start looking for word datasets that are diverse? and maybe, just maybe like add emotional cues like if suprised/curious look -> question mark etc.

Day 10: I was only able to fix the lag by changing the architecture a bit: I moved emotion detection & asl classification to background thread while keeping mp hand tracking synchronized on the main thread. Tomorrow I will: find out a way to store all past words locally so that somebody else will be able to use my data afterwards, start looking for word datasets that are diverse and  add emotional cues like if suprised/curious look -> question mark etc. Also: I'll try to fix the layout of the demo because I broke it today.

Day 11: Added 2 things: fixed the layout i broke yesterday & added emotional punctuation. cues are neutral(.) angry (! (angry)) happy (! (happy)) shocked(?). Tomorrow I will: add the option to choose between a male & a female voice & start looking for word datasets.

Day 12: I added the option to change between male & female voices. tomorrow i will add the option to change punctuation before confirming the final sentecne to change the pronunciation, make the algorithm stop reading (angry) & (happy) and finally increase accuracy to hopefully %99.

Day 13: Today I worked on a couple of things: refined my testing system (basically made it more detailed so i could see which letters were the most frequently confused). as a result, i saw that the most common mix-ups/issues were: n vs a (closed fist gestures), r vs u (crossed & parallel fingers, both 0.89% precision), j (motion sign thats represented statically) & i & j confusion. i added the puncutation selection tool where . = neutral ! = angry / happy (there’s a sub-menu that pops up if you choose !), and ? = question / suprised. i made it so that when you change the punctuation, the pitch of the voice changes. i had an issue where the pitches i entered weren’t properly in hertz but i went on to fix that. next, i made the algorithm stop reading (angry) and (happy) by removing them entirely. tomorrow: i’m going to eliminate the emotion detector, there’s no need for it anymore. try to fix the main mix-ups / problems (possibly eliminate the static problem? is there a way to make the algorithm detect things dynamically?)

Day 14: I deleted the emotions detector and refined some stuff in like 15 mins then tried to push to github for literally two hours.

Day 15: literally no time!! im writing this in a car on the way to my friends house. anyway, today i refined my train_model.py by upgrading feature extraction to hopefully minimize r/u and n/a confusion. it kinda worked but i need to do more extensive testing tomorrow. then i updated detector.py to match train_py in how they extract. tomorrow i'll try to introducce basic words.

Day 16: Today was a longer update, I fixed my train_py because everything it was detecting was -, got started on my word dataset ( I have 9 words so far), added an alphabet mode to word mode to detector.py, updated detector.py to match the new files, created two new files called collect_words and train_words - I use collect_words to record 30 clips of each sign, and then it adds them to asl_words.csv and I use train_wrods to train the model. I also updated my gitignore to not track the audio cache. Tomorrow: I plan to add roughly 20 words into the dataset, make hand tracking more accurate, and add a cooldown to the word detecting as it just spams words. Also, maybe I'll add a pop-up that pops up when the algorithm is confused between two words/letters like did you mean x or y?

Day 17:  I did add a 3 second cooldown to word detecting & made word tracking cuter, and attempted to add more words into collect_words.py but the algorithm kinda broke and it didn't save my 30 samples. tomorrow: im going to fix collect_words.py, add 10 words to it, fix Prediction error: X has 126 features, but HistGradientBoostingClassifier is expecting 123 features as input, add a did you mean x or y button and finally add a tense button like at the end when the sentence is finished there will be 3 buttons that are like: present, future, past, and the algorithm will restructure the sentence accordingly.

Day 18: Today I: fixed the prediction error, broke the hand tracking & letter spelling then fixed it, fixed the finish motion being detected while signing a word, added the did u mean x or y button initially with a confidence of 80%+ but then updated it to raw confidence of 95%+, i then fixed the progress bars that I broke. tomorrow I'm going to fix collect_words.py, add 10 words to it, fix spelling accuracy and add a tense button.

Day 19: Today I fixed the accuracy of the spelling, put the progress bars on the top left, remove did u mean x or y button (it wasn't as useful as I thought it'd be, add tense transformer button (i didn't test it yet) & added the gemini api for the tense transformer. tomorrow: i'm going to test the tense transformer & finally fix collect_words.py (& add 10 words if i am able to. and hopefully finally get around to making this readme more formal.

Day 20: I finally got around to fixing collect_words.py & added 10 words to it. I did add a type 's' to start detecting word in word mode as it just wasn't working well enough even with the buffer. Tomorrow I'm going to fix train_words.py, test detector.py, test the tense transformer, add 10 more words, add the option to delete the last sample for collect_words.py, maybe add face tracking like brow furrowing etc to collect_words.py, add the sarcastic pronounciation option to detector.py,  add the option to pick word when it detects a sign that means multiple words. so when typing which word the sign is in collect_words.py when there's slashes between two/three words, it will store those.

Day 21: today I: fixed train_words.py, tested detector.py, tested the tense transformer, added the option to delete the last sample for collect_words.py, added the sarcastic and sad pronounciation option to detector.py, made it so you have to click s for it to start detecting words in detector.py word mode, and added a start/finish button to detector.py. Tomorrow I'm going to add 5 words to my dataset, add face tracking to collect_words.py and incroporate that into detecting in detector.py, add the option to pick word when it detects a sign that means multiple words. so when typing which word the sign is in collect_words.py when there's slashes between two/three words, it will store those  , fix accuracy for words & try to make the tense transformer better ( more tenses? learn more abt asl grammar)

Day 22: I added face tracking to collect_words.py and updated train_words.py and detector.py accordingly. I then added the synonym model. I've yet to try them. Tomorrow I hope I'll have time to add a few words to the dataset, learn more abt asl grammar & make the tense transformer better, and fix the accuracy for words.

Day 23: I basically made the tense transformer better by adding present continous tense etc, sharpening up the gemini prompt and a few more things. I also fixed the accuracy for words (or atleast I tried). Tomorrow: I'll add 5-6 words to the dataset, and make the ui cuter?? I feel like from here on out I'll just chill and amass a big dataset.

Day 24: I just got a ssh key. Anyway, today I redesigned the ui to be more warm/ minimalistic. Tomorrow I'll add more words -- from here, I'm going to have shorter updates and just add more words.

Day 25: Added 6 words to the dataset. Tomorrow I will add 5-6 more, and maybe add like emotions logic as I learned that the amount of facial expression while you're signing it shows how intense you're experiencing that emotion.

Day 26: Today I added 7 words, made hand tracking more accurate & finally incorporated the emotion logic. Tomorrow I'll add more words, maybe make the UI for the collect_words.py cuter as well.

Day 27: Added 10 words, initialized git lfs for my dataset as it wasn't pushing to git. Tomorrow I'll make the ui cuter for collect_words.py and add more words.

Day 28: Added 14 words, tomorrow I'll make the ui cuter 

Day 29: Today I made the ui cuter for collect_words.py, re-imagined the emotions logic: its (e) for any emotions that are being recorded into collect_words.py, re-incorporated the progress bar for detecting in detector.py, and a few things i dont think i can remember. I also added  8 words, tomorrow I'll add words, add emotions & also maybe look into making this algorithm an app.  and add arm tracking to both detector.py & to collect_words.py

Day 30: Added arm tracking, tomorrow I'll look into making this an app & adding a few words.

Day 31: I added words & emotions today! Tomorrow I'll add more words.

Day 32: Added more words, tomorrow I'll add words again & try to figure out git lfs as I'm unable to push the updated models. I also updated collect_words.py and train_words.py today.

Day 33: I'll try to add multiple sign for one word logic and more words every other day. Today I added a lot of words.

Day 34 (after a long break): I added 2+ signs for one word logic. In ASL, there are often more than one sign for a word, much like english. My prior version didn't support this logic so I updated collect_words.py, detector.py and train_words.py

Day 35: Today I added 7 words ( I cant push them to git as my model is too large & the dataset is, too). I made collect_words.py pop up a confirmation when the user presses d to delete. The options are yes/no

Day 36: I got this message while I was adding words ning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon. Tomorrow I'll fix my code according to this. Plus, I'll add the option to change the word in collect_words.py before the collecting process begins.

Day 37: Added 5 words today. Testing showed great accuracy but it uses the 'very' logic for even non e words. Extremely laggy and shows sign even in word mode. Will fix these.

Day 38: I added 4 words -- we're officialy at 100 words in total! I fixed the deprecating thing in collect_words.py. Tomorrow I'll add the option to change the word in collect_words.py when the window opens.

Day 39: Optimized detector.py for less lag will fix these tomorrow: cant detect hands well, sign displays even in word mode, very logic incorporated even into words, fix color scheme according to color blind ones, fix controls for start presss start or s v_2 displayed even when not supposed to

Day 40: Fixed hand tracking & removed alphabet detection in word mode. Tomorrow I'm going to fix: Very logic incorporated into words, fix color scheme according to color blind adaptation, fix controls for start (add s option), make space detection better, switch out "stop" for "end", fix _v2 being displayed in multi sign words

Day 41: My computer has crashed due to detector.py's lag. It wont run any files, and it won't even save detector.py, I'll figure it out tomorrow.

Day 42: I have, in fact, figured it out. Today I added live transcription. The algorithm can now switch between transcription and signing. I did this to fully ensure that both communities, hearing and hard-of-hearing/deaf, have a way to communicate with eachother. Redesigned UI in its entirety & made colors more aesthetic. I cleaned up the UI towards the end of the session, removed palm start because it was really faulty, fixed the voice bug, added a delete & edit button to transcription mode, and made s start whatever mode you're on. This session alone kind of finished my personal project. I also did my first ever intensive testing, which led me to see that my computer heats up a lot after intense use. I'll need to optimize my code soon.

Day 43: I fixed the very logic. Previously, it was adding the emotions logic (adding the very prefix even though no _e). Now it doesn't do that and only applies it to words ending with _e. I fixed the algorithm displaying _v1 & _v2 (the logic for two/three+ signs for one word) by making it check if the word has _e and if it does, just stripping _v1 etc. my computer heats up after running detector.py a bit, as aformenetioned and today the key optimizations i made are: 
- webcam frames are now downscaled before passing to mp models.
- pause vision models in transcription.
- face mesh and pose models now sample every 3 frames (frame_counter % 3 == 0)
- added a micro sleep target frame limiter so the cpu doesnt spin at %100 capacity when the cameras idel

Day 44: I added 7 words & updated my main ASL words model. Tomorrow I'm going to look into deploying this.

Day 45: Added 6 words & wrote my personal project report.

Day 46: Added 6 words.

Day 47: Added 7 words.

Day 48: Added 7 words. Mostly colors.
