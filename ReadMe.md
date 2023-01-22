## Sentiment  Analyzer IMDB reviews Dataset
This model predicts whether a movie review is positive or negative based on sentiments by using LSTM, a recurrent neural network model. This model achieved an accuracy of over 90% for a buffer size of 10000 samples and epochs 20.

## IMDB datasdet:
This is a dataset of 50,000 movies reviews from IMDB, labeled by sentiment (1-positive/0-negative). Reviews have been preprocessed, and each review is encoded as a list of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".
As a convention, "0" does not stand for a specific word, but instead is used to encode the pad token.

## Output:

### Positive sentiment:1
### Negative sentiment:0

### Sample input text-1:

b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it."
### Label : 0

### Sample input text-2:  

[b'As someone who has read the book, I can say that this is vastly inferior to the big American version starring Gwyneth Paltrow. There are various reasons for this. Firstly, Emma is too unpleasant. Yes, she has faults, and isn\'t the easiest person to like - but the viewer shouldn\'t downright start to despise her. Secondly, Mr Knightly is miscast. His brooding and melancholy in this version are better suited to a Bronte or Gaskell adaptation than Austen, and throw the mood of the whole affair "off". Thirdly, Samantha Morton is too strong an actress to be relegated to the role of Harriet; and why was she made to look so sickly? Harriet is supposed to be blonde and blooming - not to look as if she\'s going to be carried off by consumption in the next scene. Fourthly, the structure has been mucked up and scenes cut. At the end, when Emma decides she loves Mr Knightly, it comes across as utterly baffling because this narrative hasn\'t been adequately shown and carried along throughout the film. Fifthly, what was going on, exactly, with Mrs Elton\'s accent? She went from sounding like an American actress trying to suppress her own accent at the beginning, to all out American half-way through, and then back to English at the end. Finally, this dragged at the end. The book and the big film version end with the wedding of Emma and Mr Knightly. This version drags on confusingly after the announcement of the wedding without actually showing us the ceremony.<br /><br />All in all, a rather haphazard attempt. Read the book or rent the Paltrow version instead'
 b"I loved October Sky. The thing I loved most had to be the music. It worked two ways: in the first hour of the film, it gives the viewer a time-frame. This is done by playing songs from the late Fifties. In the second hour, an instrumental score takes over. The music now fits the mood of the film perfectly.<br /><br />I did not only enjoy the music, I also quite enjoyed the cast. Jake Gyllenhaal as Homer Hickam was especially a surprise for me. He gave off a first-class performance, as did Chris Owen (Quentin) and Chris Cooper (John Hickam).<br /><br />I've seen this movie about escaping the life already laid out for you twice now, and both times I thoroughly enjoyed myself."
 b'with this ABC family attempt of the hit blockbuster "cheaper by the dozen" comes an obnoxious amount of corny dialogue, shallow plot lines, and cheesy comebacks. With about two good actors among many wanna-be\'s, this movie was a major disappointment. Its a Hollywood-wannabe ditto of an already bad plot. Then, because they needed a lot of actors, that meant that they\'d probably be more lenient. So the acting wasn\'t five-star. The plot moved fairly fast, and the twists were bad and had horrible timing. The junction of characters and the "end relationships" were also too mushy and clich\xc3\xa9d for me. Spare yourself and rent something better.']
### Label: [0 1 0]

### Sample input text-3: 

[b'So why does this show suck? Unfortunately, that really is the only question, because there is no doubt that it does.<br /><br />For those unfamiliar with the premise of the show, the doomed-to-be-shortlived series Cavemen focuses on a number of Neanderthals and their struggle to exist in modern day America and is based on the characters featured in a series of television ads for Geico Insurance. The concept is solid and there is every reason to think it could be executed successfully.<br /><br />I had to think about it for awhile, but then the tagline from the commercials -- something to the effect of "We\'re not that much different from you" provided me with the key to the show\'s suckiness. Even though cavemen/Neanderthals are actually a different species than humanity, the title characters of this show, it turns out, are exactly the same as those of us who are boring jerks.<br /><br />Maybe its my background as a game writer -- rather than a soulless, hack, committee-based writer from California -- but this show had so much potential, and none of it has been realized. To start with, the producers should have focused on the fun things that would make cavemen different from us.<br /><br />What could conceivably be funny, for example, about giving them occupations like perpetual grad student and furniture store clerk, when they would have more compellingly been drawn to things like subterranean utility workers and guides at cave parks? Why would they play prosaic games like squash, when a whole episode could be devoted to them trying get hunting licenses to go after game with spears? A show like this could write itself, and it takes some willfully bad writing to make it quite so crappy and boring.<br /><br />Another tiresome aspect of this show is an attempt to portray the cavemen as being subjected to a number of stereotypes associated with various human minorities. Yawn! This has been done so many times before, and never more drearily than this. And, as noted previously, Neanderthals really are a different species, so using them as a metaphor for racial stereotyping is both uncompelling and off the mark.<br /><br />Responses are welcome, including those from anyone who wants to tell me why I\'m wrong. I\'d like to enjoy this show and am just sorry that I have thus far been unable to.<br /><br />Michael J. Varhola, Skirmisher Online Gaming Magazine'
b"City Hall takes on the politics of a city rather than country, state or any sort of major political table. Granted it shines on New York City which is a huge political arena, especially nowadays, but it still goes for a smaller scale and puts the microscope on a few key players in a city wide scandal stumbled on by the mayor's right hand. Director Harold Becker is a director very familiar with elements of the thriller having done Mercury Rising, Malice, and Domestic Disturbance and I think in many ways he incorporates so many of the formulaic thriller genre that its almost to a fault. I mean City Hall is meant to be a political drama, not a thriller but instead when all is said and done and once you get to the meat and potatoes of the film it feels and looks like a thriller but a decent one at that with very important part of the recipe that immediately makes it stand out...what else...or rather who else...Al Pacino. The film begins by giving you a really good look at life in the mayoral office and the inner workings of the city. As the film continues it broadens its political spectrum to include a democratic boss, and his connections and then we are introduced to some of the goings on within the city. As events unfold a mystery begins and the political aspect is kind of left in the background but it still has a brilliant set up.<br /><br />I absolutely hate talking about Al Pacino. I mean even if ONCE he didn't give a good performance how could I ever say it? The man is acting royalty. There is just something brilliant about his entire demeanor. In City Hall Pacino plays the New York City mayor. He has a sense of duty and honor and immediately appears to be a very upstanding politician. He also delivers one of the most powerful and outright engaging speeches I've ever seen at the 'James Bone' Funeral. I re-watched that speech four times and the first time I watched Pacino give it, my mouth gaped open and I almost wanted to stand up and applaud. Its brilliantly written and brilliantly delivered by Pacino. John Cusack, who I really do enjoy as an actor, turns in a mediocre and overdone performance as the deputy Mayor Kevin Calhoun. He is kind of the focus of the film and him and Pacino have good chemistry together when they are on screen but there is just something in this performance...he seems like he's trying too hard. His accent is just bizarre, and although he is supposed to be cutthroat and intimidating he doesn't get seem to pull it off. Maybe he was having an off film. Bridget Fonda, on her way out of her high point stardom does an okay job as attorney for police widows Marybeth Cogan. Her performance is very similar to Cusack's in that she just doesn't seem to find her groove with this character. Danny Aiello is terrific although his character is a little under explored as democratic boss with ties to the mafia Frank Anselmo. Martin Landau makes a decent cameo as Judge under scrutiny Walter Stern.<br /><br />The problem with City Hall is evident in my review of the characters and actors. Everyone is...okay. There is a lot of back story that they try to bring out without actually showing it and it unfortunately leaves you just a little bit confused about the whole conspiracy. And of course you have Al Pacino in a rather small supporting role but he's absolutely brilliant at it and outshines and overshadows every other actor in the film. It almost feels like maybe they are intimidated by him being on screen with them. So City Hall could have been this huge political epic drama/thriller but it felt cut and toned down to an average run of the mill one BUT it still has to be seen for Pacino and a different spin on the inner working of politics. If you just won't see this movie than find Pacino's speech at James Bone funeral because the word electrifying doesn't seem to give it justice but you can see what makes Al Pacino so incredible because in a mediocre film he pulls out this wallop of a speech and makes you feel it. If you're a John Cusack fan which I am...he's definitely done better but he is the main character and all in all he does get his justice. A decent movie but unfortunately potential loss. 7.5/10"
b'...from this awful movie! There are so many things wrong with this film, acting, writing, direction, editing, etc. that it\'s amazing that something rises to the top and proves itself to be the absolute worst. The music! I noted that the film has two composers listed. This must be the reason why every single frame has music, of the absolute worst "D" movie style drivel. They have never heard of the expression "less is more". It got so painful to listen to, I muted the sound every time there was no dialogue, not that the dialogue was that good. You have to feel sorry for Robert Wagner and Tom Bosley, I\'m sure they didn\'t see roles like this in the twilight of their careers. See it at your own risk.']
### Label: [0 1 0]

## Sample 50 Word tokens

['' '[UNK]' 'the' 'and' 'a' 'of' 'to' 'is' 'in' 'it' 'i' 'this' 'that'
 'br' 'was' 'as' 'for' 'with' 'movie' 'but' 'film' 'on' 'not' 'you' 'are'
 'his' 'have' 'he' 'be' 'one' 'its' 'at' 'all' 'by' 'an' 'they' 'from'
 'who' 'so' 'like' 'her' 'just' 'or' 'about' 'has' 'if' 'out' 'some'
 'there' 'what'] 
 
 ## Model performance
 <img width="987" alt="Accuracy" src="https://user-images.githubusercontent.com/122699563/213945011-ad4f55e8-f6b8-43ab-8df6-b198c3a51d88.png">

 

