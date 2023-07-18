WARNING_TEXT = """If no entities are presented in any categories or If you don't know or understand or not sure,keep it "O". 
Split all words into space. "B" stands for Begin,which is the beginning of the object name. "I" stands for Inside,which is the internal part of the object name.
All words EXCEPT "O" MUST start with a "B-",but if they are separated by a spacebar and are still a single word,start the first word with a "B-" and the next word with an "I-". There is NO "B-O" or "I-O". 
If you think that provided sentence is incomplete,Just Answer the entities like example I provided. And Please Answer ONLY example sentence's Output and NEVER answer ANYTHING other than output."""

CONLL_BASELINE_PROMPT = f"""An entity is a object in the world like a place or person and a named entity is a phrase that uniquely refers to an object by its proper name (Hillary Clinton), acronym (IBM), nickname (Opra) or abbreviation (Minn.).
ONLY return entities DESCRIBED after.
Entity DESCRIPTIONS are defined as follows:
1. PER: Person (PER) entities are limited to humans (living, deceased, fictional, deities, ...) identified by name, nickname or alias. Don’t include titles or roles (Ms., President, coach). Include suffix that are part of a name (e.g., Jr., Sr. or III).
2. MISC: Any format of miscellaneous.
3. LOC: Location (LOC) entities include names of politically or geographically defined places (cities, provinces, countries, international regions, bodies of water, mountains, etc.). Locations also include man-made structures like airports, highways, streets, factories and monuments.
4. ORG: Organization (ORG) entities are limited to corporations, institutions, government agencies and other groups of people defined by an established organizational structure. Some examples are businesses (Bridgestone Sports Co.), stock ticker symbols (NASDAQ), multinational organizations (European Union), political parties (GOP) non-generic government entities (the State Department), sports teams (the Yankees), and military groups (the Tamil Tigers). Do not tag ‘generic’ entities like “the government” since these are not unique proper names referring to a specific ORG. 
{WARNING_TEXT}
"""

CONLL_PROMPT = f"""{CONLL_BASELINE_PROMPT}
This data from news stories and articles consists of eight files covering two languages: English and German. Please Concentrate on four types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
\nExamples:
1. Sentence: ['Mr. Jacob','lives','in','Madrid','since','12th January 2015','(','2015-01-12',')','.']
Output:\nMr. Jacob&&B-PER\nlives&&O\nin&&O\nMadrid&&B-LOC\nsince&&O\n12th January 2015&&B-MISC\n(&&O\n2015-01-12&&B-MISC\n)&&O\n.&&O\n\n
2. Sentence: ['that','is','to','end','the','state','of','hostility',',','\'','Thursday',''s','overseas','edition','of','the','People',''s','Daily','quoted','Tang','as','saying','.']
Output:\nthat&&O\nis&&O\nto&&O\nend&&O\nthe&&O\nstate&&O\nof&&O\nhostility&&O\n,&&O\n"&&O\nThursday&&O\n's&&O\noverseas&&O\nedition&&O\nof&&O\nthe&&O\nPeople&&B-ORG\n's&&I-ORG\nDaily&&I-ORG\nquoted&&O\nTang&&B-PER\nas&&O\nsaying&&O\n.&&O\n\n"""

KLUE_PROMPT = f"""An entity is a object in the world like a place or person and a named entity is a phrase that uniquely refers to an object by its proper name (Hillary Clinton), acronym (IBM), nickname (Opra) or abbreviation (Minn.).
ONLY return entities DESCRIBED after.
Entity DESCRIPTIONS are defined as follows:
1. PS (Person): Name of an individual or a group
2. LC (Location): Name of a district/province or a geographical location
3. OG (Organization): Name of an organization or an enterprise
4. DT (Date): Expressions related to date/period/era/age
5. TI (Time): Expressions related to time
6. QT (Quantity): Expressions related to quantity or number including units
{WARNING_TEXT}
\nExample:
1. Sentence: ['영','동','고','속','도','로',' ','강','릉',' ','방','향',' ','문','막','휴','게','소','에','서',' ','만','종','분','기','점','까','지',' ','5','㎞',' ','구','간','에','는',' ','승','용','차',' ','전','용',' ','임','시',' ','갓','길','차','로','제','를',' ','운','영','하','기','로',' ','했','다','.']
Output:\n영&&B-LC\n동&&I-LC\n고&&I-LC\n속&&I-LC\n도&&I-LC\n로&&I-LC\n &&O\n강&&B-LC\n릉&&I-LC\n &&O\n방&&O\n향&&O\n &&O\n문&&B-LC\n막&&I-LC\n휴&&I-LC\n게&&I-LC\n소&&I-LC\n에&&O\n서&&O\n &&O\n만&&B-LC\n종&&I-LC\n분&&I-LC\n기&&I-LC\n점&&I-LC\n까&&O\n지&&O\n &&O\n5&&B-QT\n㎞&&I-QT\n &&O\n구&&O\n간&&O\n에&&O\n는&&O\n &&O\n승&&O\n용&&O\n차&&O\n &&O\n전&&O\n용&&O\n &&O\n임&&O\n시&&O\n &&O\n갓&&O\n길&&O\n차&&O\n로&&O\n제&&O\n를&&O\n &&O\n운&&O\n영&&O\n하&&O\n기&&O\n로&&O\n &&O\n했&&O\n다&&O\n.&&O\n\n
2. Sentence: ['중','국',' ','후','난','(','湖','南',')','성',' ','창','샤','(','長','沙',')','시',' ','우','자','링','(','五','家','岭',')','가',' ','한',' ','시','장','에','서',' ','1','4','일',' ','오','전',' ','1','0','시',' ','1','5','분','께',' ','칼','부','림',' ','사','건','이',' ','일','어','나',' ','5','명','이',' ','숨','지','고',' ','1','명','이',' ','부','상','했','다','고',' ','중','신','넷','이',' ','1','4','일',' ','보','도','했','다','.']
Output:\n중&&B-LC\n국&&I-LC\n &&I-LC\n후&&I-LC\n난&&I-LC\n(&&O\n湖&&I-LC\n南&&I-LC\n)&&O\n성&&I-LC\n &&I-LC\n창&&I-LC\n샤&&I-LC\n(&&O\n長&&I-LC\n沙&&I-LC\n)&&O\n시&&I-LC\n &&I-LC\n우&&I-LC\n자&&I-LC\n링&&I-LC\n(&&O\n五&&I-LC\n家&&I-LC\n岭&&I-LC\n)&&O\n가&&I-LC\n &&O\n한&&O\n &&O\n시&&O\n장&&O\n에&&O\n서&&O\n &&O\n1&&B-DT\n4&&I-DT\n일&&I-DT\n &&O\n오&&B-TI\n전&&I-TI\n &&I-TI\n1&&I-TI\n0&&I-TI\n시&&I-TI\n &&I-TI\n1&&I-TI\n5&&I-TI\n분&&I-TI\n께&&O\n &&O\n5&&B-QT\n명&&I-QT\n이&&O\n &&O\n숨&&O\n지&&O\n고&&O\n &&O\n1&&B-QT\n명&&I-QT\n이&&O\n &&O\n부&&O\n상&&O\n했&&O\n다&&O\n고&&O\n &&O\n중&&B-OG\n신&&I-OG\n넷&&I-OG\n이&&O\n &&O\n1&&B-DT\n4&&I-DT\n일&&I-DT\n &&O\n보&&O\n도&&O\n했&&O\n다&&O\n.&&O\n
"""
#  Please Answer ONLY example sentence's Output and NEVER answer ANYTHING other than output.
# even if provided sentence is incomplete,Just Answer the entities like example I provided. And
# Choose to tag in character level. First of all,whitespace-split units (eojeols) are often not a single word and are a composite of content words and functional words (e.g.,‘담주가 (the next week is)’ = ‘담주 (the next week)’ + ‘가 (is)’). Second,many compound words in Korean contain whitespaces.

base_request_data = [("system",
                      "You are a smart and intelligent Named Entity Recognition (NER) system. The task is to labelling entities. I will provide you the definition of the entities you need to extract,the sentence from where your extract the entities and the output format with examples."),
                     ("user", "Are you clear about your role?"),
                     ("assistant",
                      "Sure,I'm ready to help you with your NER task. Please provide me with the sentence to get started.")]

KLUE_KOREAN_PROMPT = """다음에 설명된 엔티티만 반환합니다.
엔티티 설명:
1. PS(개인): 개인 또는 그룹의 이름
2. LC(위치): 지역/주 또는 지리적 위치 이름
3. OG(조직): 조직 또는 기업 이름
4. DT(날짜): 날짜/기간/시대/연령과 관련된 표현식
5. TI(시간): 시간 관련 표현식
6. QT(수량): 단위를 포함한 수량 또는 수와 관련된 표현
어떤 범주에도 제시된 엔티티가 없거나, 이해하지 못하거나, 확실하지 않은 경우 "O"로 유지합니다.
"B"는 개체 이름의 시작 부분인 Begin의 약자입니다. "I"는 개체 이름의 내부 부분인 내부를 나타냅니다.
"O"를 제외한 모든 단어는 "B-"로 시작해야 하지만 스페이스바로 구분되어 있고 여전히 한 단어인 경우 첫 단어는 "B-"로 시작하고 다음 단어는 "I-"로 시작합니다. "B-O" 또는 "I-O"는 없는 엔티티이므로 절대 출력하지 마세요. 
또한, 예시 문장의 출력에만 답하고 아무 것도 답하지 마세요.
\n예제:
1. 문장: ['영','동','고','속','도','로',' ','강','릉',' ','방','향',' ','문','막','휴','게','소','에','서',' ','만','종','분','기','점','까','지',' ','5','㎞',' ','구','간','에','는',' ','승','용','차',' ','전','용',' ','임','시',' ','갓','길','차','로','제','를',' ','운','영','하','기','로',' ','했','다','.']
출력결과:\n영&&B-LC\n동&&I-LC\n고&&I-LC\n속&&I-LC\n도&&I-LC\n로&&I-LC\n &&O\n강&&B-LC\n릉&&I-LC\n &&O\n방&&O\n향&&O\n &&O\n문&&B-LC\n막&&I-LC\n휴&&I-LC\n게&&I-LC\n소&&I-LC\n에&&O\n서&&O\n &&O\n만&&B-LC\n종&&I-LC\n분&&I-LC\n기&&I-LC\n점&&I-LC\n까&&O\n지&&O\n &&O\n5&&B-QT\n㎞&&I-QT\n &&O\n구&&O\n간&&O\n에&&O\n는&&O\n &&O\n승&&O\n용&&O\n차&&O\n &&O\n전&&O\n용&&O\n &&O\n임&&O\n시&&O\n &&O\n갓&&O\n길&&O\n차&&O\n로&&O\n제&&O\n를&&O\n &&O\n운&&O\n영&&O\n하&&O\n기&&O\n로&&O\n &&O\n했&&O\n다&&O\n.&&O\n\n
2. 문장: ['중','국',' ','후','난','(','湖','南',')','성',' ','창','샤','(','長','沙',')','시',' ','우','자','링','(','五','家','岭',')','가',' ','한',' ','시','장','에','서',' ','1','4','일',' ','오','전',' ','1','0','시',' ','1','5','분','께',' ','칼','부','림',' ','사','건','이',' ','일','어','나',' ','5','명','이',' ','숨','지','고',' ','1','명','이',' ','부','상','했','다','고',' ','중','신','넷','이',' ','1','4','일',' ','보','도','했','다','.']
출력결과:\n중&&B-LC\n국&&I-LC\n &&I-LC\n후&&I-LC\n난&&I-LC\n(&&O\n湖&&I-LC\n南&&I-LC\n)&&O\n성&&I-LC\n &&I-LC\n창&&I-LC\n샤&&I-LC\n(&&O\n長&&I-LC\n沙&&I-LC\n)&&O\n시&&I-LC\n &&I-LC\n우&&I-LC\n자&&I-LC\n링&&I-LC\n(&&O\n五&&I-LC\n家&&I-LC\n岭&&I-LC\n)&&O\n가&&I-LC\n &&O\n한&&O\n &&O\n시&&O\n장&&O\n에&&O\n서&&O\n &&O\n1&&B-DT\n4&&I-DT\n일&&I-DT\n &&O\n오&&B-TI\n전&&I-TI\n &&I-TI\n1&&I-TI\n0&&I-TI\n시&&I-TI\n &&I-TI\n1&&I-TI\n5&&I-TI\n분&&I-TI\n께&&O\n &&O\n5&&B-QT\n명&&I-QT\n이&&O\n &&O\n숨&&O\n지&&O\n고&&O\n &&O\n1&&B-QT\n명&&I-QT\n이&&O\n &&O\n부&&O\n상&&O\n했&&O\n다&&O\n고&&O\n &&O\n중&&B-OG\n신&&I-OG\n넷&&I-OG\n이&&O\n &&O\n1&&B-DT\n4&&I-DT\n일&&I-DT\n &&O\n보&&O\n도&&O\n했&&O\n다&&O\n.&&O\n"""

TWEE_PROMPT = f"""{CONLL_BASELINE_PROMPT}\nExamples:
1. Sentence: ["RT", "@USER1502", "Jesus", "!", "Now", "playing", "at", "Dunkin", "Donuts", "Travie", "McCoy", ":", "Billionaire", "ft.", "Bruno", "Mars", "URL123", "January", "16", ",", "2011"]
Output:\nRT&&O\n@USER1502&&O\nJesus&&B-PER\n!&&O\nNow&&O\nplaying&&O\nat&&O\nDunkin&&B-LOC\nDonuts&&I-LOC\nTravie&&B-PER\nMcCoy&&I-PER\n:&&O\nBillionaire&&B-MISC\nft.&&O\nBruno&&B-PER\nMars&&I-PER\nURL123&&O\nJanuary&&O\n16&&O\n,&&O\n2011&&O\n\n
2. Sentence: ["CNS", "News", ":", "Apple", "Now", "Accepting", "Design", "Award", "Nominations", "–", "PC", "Magazine", "URL124"]
Output:\nCNS&&B-ORG\nNews&&I-ORG\n:&&O\nApple&&B-ORG\nNow&&O\nAccepting&&O\nDesign&&B-MISC\nAward&&I-MISC\nNominations&&O\n–&&O\nPC&&B-ORG\nMagazine&&I-ORG\nURL124&&O\n\n"""

BTC_PROMPT = f"""
An entity is a object in the world like a place or person and a named entity is a phrase that uniquely refers to an object by its proper name (Hillary Clinton), acronym (IBM), nickname (Opra) or abbreviation (Minn.).
All Sentence comes from Twitter, sampled across different regions, temporal periods, and types of Twitter users. And collected in order to capture temporal, spatial and social diversity.
ONLY return entities DESCRIBED after. NEVER output entities other than DESCRIPTIONS.
Entity DESCRIPTIONS are defined as follows:
1. PER: Short name or full name of a person from any geographic regions.
2. LOC: Name of any geographic location,like cities,countries,continents,districts etc.
3. ORG: a group of people who work together in an organized way for a shared purpose.
{WARNING_TEXT}
\nExamples:
1. Sentence: [ "Gene", "Cohen", "'s", "Beady", "Eye", "have", "already", "covered", "the", "High", "Flying", "Birds", "'", "album", "." ]
Output:\nGene&&B-PER\nCohen&&I-PER\n's&&O\nBeady&&B-ORG\nEye&&I-ORG\nhave&&O\nalready&&O\ncovered&&O\nthe&&O\nHigh&&B-ORG\nFlying&&I-ORG\nBirds&&I-ORG\n'&&O\nalbum&&O\n.&&O\n\n
2. Sentence: [ "Potters", "Bar", "is", "in", "Hertfordshire", "(", "Central", "London", ")",  ".", "http://t.co/IkncHGtI" ]
Output:\nPotters&&B-LOC\nBar&&I-LOC\nis&&O\nin&&O\nHertfordshire&&B-LOC\n(&&O\nCentral&&B-LOC\nLondon&&I-LOC\n)&&O\n.&&O\nhttp://t.co/IkncHGtI&&O\n\n
"""

# https://huggingface.co/datasets/wnut_17
WNUT_PROMPT = f"""
An entity is a object in the world like a place or person and a named entity is a phrase that uniquely refers to an object by its proper name (Hillary Clinton), acronym (IBM), nickname (Opra) or abbreviation (Minn.).
Sentences comes from Reddit, Twitter, YouTube, and StackExchange comments. 
The goal of this task is to provide a definition of emerging and of rare entities, and based on that, also datasets for detecting these entities.
ONLY return entities DESCRIBED after.
Entity DESCRIPTIONS are defined as follows:
1. person: Don’t mark people that don’t have their own name. Include punctuation in the middle of names. Fictional people can be included, as long as they’re referred to by name (e.g. Harry Potter).
2. location: Name of any geographic location,like cities,countries,continents,districts etc.
3. corporation: Names of corporations (e.g. Google). Don’t mark locations that don’t have their own name. Include punctuation in the middle of names.
4. product: Name of products (e.g. iPhone). Don’t mark products that don’t have their own name. Include punctuation in the middle of names. Fictional products can be included, as long as they’re referred to by name (e.g. Everlasting Gobstopper). It’s got to be something you can touch, and it’s got to be the official name.
5. creative-work: Names of creative works (e.g. Bohemian Rhapsody). Include punctuation in the middle of names. The work should be created by a human, and referred to by its specific name.
6. group: Names of groups (e.g. Nirvana, San Diego Padres). Don’t mark groups that don’t have a specific, unique name, or companies (which should be marked corporation).
{WARNING_TEXT}
\nExamples:
1. Sentence: ["RT", "@DamnTeenQuotes", "#BattlestarGalactica", "'s", "32nd", "Anniversary", "#StarWars", "#TheCloneWars", ".", "going", "to", "alderwood", "again"]
Output:\nRT&&O\n@DamnTeenQuotes&&O\n#BattlestarGalactica&&B-creative-work\n's&&O\n32nd&&O\nAnniversary&&O\n#StarWars&&B-creative-work\n#TheCloneWars&&I-creative-work\n.&&O\ngoing&&O\nto&&O\nalderwood&&B-location\nagain&&O\n\n
2. Sentence: ['we', 'got', 'cody', "'s", 'ipod', 'I', 'remember', 'when', 'i', 'was', 'your', 'age', ',', 'spencer', 'from', 'iCarly', 'was', 'Crazy', 'Steve', ',', 'Carly', 'was', 'Megan', 'and', 'Josh', 'was', 'fat', '.', '#damnteenquotes']
Output:\nwe&&O\ngot&&O\ncody&&O\n's&&O\nipod&&B-product\nI&&O\nremember&&O\nwhen&&O\ni&&O\nwas&&O\nyour&&O\nage&&O\n,&&O\nspencer&&B-person\nfrom&&O\niCarly&&B-creative-work\nwas&&O\nCrazy&&B-person\nSteve&&I-person\n,&&O\nCarly&&B-person\nwas&&O\nMegan&&B-person\nand&&O\nJosh&&B-person\nwas&&O\nfat&&O\n.&&O\n#damnteenquotes&&O\n\n
"""

WIKIANN_PROMPT = f"""All Sentences comes from Wikipedia.
ONLY return entities DESCRIBED after. NEVER output entities other than DESCRIPTIONS.
Entity DESCRIPTIONS are defined as follows:
1. PER: Person (PER) entities are limited to humans (living, deceased, fictional, deities, ...) identified by name, nickname or alias. Don’t include titles or roles (Ms., President, coach). Include suffix that are part of a name (e.g., Jr., Sr. or III).
2. LOC: Location (LOC) entities include names of politically or geographically defined places (cities, provinces, countries, international regions, bodies of water, mountains, etc.). Locations also include man-made structures like airports, highways, streets, factories and monuments.
3. ORG: Organization (ORG) entities are limited to corporations, institutions, government agencies and other groups of people defined by an established organizational structure. Some examples are businesses (Bridgestone Sports Co.), stock ticker symbols (NASDAQ), multinational organizations (European Union), political parties (GOP) non-generic government entities (the State Department), sports teams (the Yankees), and military groups (the Tamil Tigers). Do not tag 'generic' entities like "the government" since these are not unique proper names referring to a specific ORG.
{WARNING_TEXT}
\nExamples:
1. Sentence: ["George", "Randolph", "Hearst", ",", "Jr", "." , "works", "at", "Zina", "Garrison-Jackson", "in", "Conch", "Key", ",", "Florida", "he", "loves", "Fireball", "Cinnamon", "Whisky" ]
Output:\nGeorge&&B-PER\nRandolph&&I-PER\nHearst&&I-PER\n,&&I-PER\nJr&&I-PER\n.&&I-PER\nworks&&O\nat&&O\nZina&&B-ORG\nGarrison-Jackson&&I-ORG\nin&&O\nConch&&B-LOC\nKey&&I-LOC\n,&&I-LOC\nFlorida&&I-LOC\nhe&&O\nloves&&O\nFireball&&B-ORG\nCinnamon&&I-ORG\nWhisky&&I-ORG\n\n
2. Sentence: ["Antiochus", "III", "of", "Commagene", "(", "died", "17", ")", ",", "reigned", "12", "BC-17", "He", "was", "High", "Sheriff", "of", "Suffolk", "from", "1670", "to", "1671", "."]
Output:\nAntiochus&&B-PER\nIII&&I-PER\nof&&I-PER\nCommagene&&I-PER\n(&&O\ndied&&O\n17&&O\n)&&O\n,&&O\nreigned&&O\n12&&O\nBC-17&&O\nHe&&O\nwas&&O\nHigh&&B-PER\nSheriff&&I-PER\nof&&I-PER\nSuffolk&&I-PER\nfrom&&O\n1670&&O\nto&&O\n1671&&O\n.&&O\n\n
"""
