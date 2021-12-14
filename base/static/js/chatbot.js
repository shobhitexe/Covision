
      
const dotProduct = (xs, ys) => {
  const sum = xs => xs ? xs.reduce((a, b) => a + b, 0) : undefined;

  return xs.length === ys.length ?
  sum(zipWith((a, b) => a * b, xs, ys))
  : undefined;
}
    
const zipWith =
    (f, xs, ys) => {
      const ny = ys.length;
      return (xs.length <= ny ? xs : xs.slice(0, ny))
          .map((x, i) => f(x, ys[i]));
    }
    
document.onload()
{
  console.log('hiii')
}        
var getBotResponse = function(query)
{
    use.load().then(model => {
      const data = [
        {"Context":"What is a coronavirus?","Answer":"Coronaviruses are a large family of viruses which may cause illness in animals or humans.�"},
        {"Context":"What is a coronavirus?","Answer":"In humans, several coronaviruses are known to cause respiratory infections ranging from the common cold to more severe diseases"},
        {"Context":"What is COVID-19?","Answer":"COVID-19 is the infectious disease caused by the most recently discovered coronavirus. This new virus and disease were unknown before the outbreak began in Wuhan, China, in December 2019."},
        {"Context":"What are the symptoms of COVID-19?","Answer":"The most common symptoms of COVID-19 are fever, tiredness, and dry cough. Some patients may have aches and pains, nasal congestion, runny nose, sore throat or diarrhea. These symptoms are usually mild and begin gradually."},
        {"Context":"What are the symptoms of COVID-19?","Answer":"Some people become infected but don�t develop any symptoms and don't feel unwell. Most people (about 80%) recover from the disease without needing special treatment."}
        ,
        {"Context":"What are the symptoms of COVID-19?","Answer":"Around 1 out of every 6 people who gets COVID-19 becomes seriously ill and develops difficulty breathing.�"}
        ,
        {"Context":"What are the symptoms of COVID-19?","Answer":"Older people, and those with underlying medical problems like high blood pressure, heart problems or diabetes, are more likely to develop serious illness. People with fever, cough and difficulty breathing should seek medical attention."},
        {"Context":"How does COVID-19 spread?","Answer":"People can catch COVID-19 from others who have the virus. The disease can spread from person to person through small droplets from the nose or mouth which are spread when a person with COVID-19 coughs or exhales.�"}
        ,
        {"Context":"How does COVID-19 spread?","Answer":"People can also catch COVID-19 if they breathe in droplets from a person with COVID-19 who coughs out or exhales droplets. This is why it is important to stay more than 1 meter (3 feet) away from a person who is sick."},
        {"Context":"Can the virus that causes COVID-19 be transmitted through the air?","Answer":"Studies to date suggest that the virus that causes COVID-19 is mainly transmitted through contact with respiratory droplets rather than through the air.�"}
        ,
        {"Context":"Can CoVID-19 be caught from a person who has no symptoms?","Answer":"The main way the disease spreads is through respiratory droplets expelled by someone who is coughing. The risk of catching COVID-19 from someone with no symptoms at all is very low."},
        {"Context":"Can CoVID-19 be caught from a person who has no symptoms?","Answer":"Many people with COVID-19 experience only mild symptoms. This is particularly true at the early stages of the disease. It is therefore possible to catch COVID-19 from someone who has, for example, just a mild cough and does not feel ill."},
        {"Context":"Can I catch COVID-19 from the feces of someone with the disease?","Answer":"The risk of catching COVID-19 from the feces of an infected person appears to be low. While initial investigations suggest the virus may be present in feces in some cases, spread through this route is not a main feature of the outbreak."},
        {"Context":"What can I do to protect myself and prevent the spread of disease?","Answer":"Regularly and thoroughly clean your hands with an alcohol-based hand rub or wash them with soap and water."},
        {"Context":"What can I do to protect myself and prevent the spread of disease?","Answer":"Maintain at least 1 metre (3 feet) distance between yourself and anyone who is coughing or sneezing."},
        {"Context":"What can I do to protect myself and prevent the spread of disease?","Answer":"Avoid touching eyes, nose and mouth."},
        {"Context":"What can I do to protect myself and prevent the spread of disease?","Answer":"Make sure you, and the people around you, follow good respiratory hygiene. This means covering your mouth and nose with your bent elbow or tissue when you cough or sneeze. Then dispose of the used tissue immediately."},
        {"Context":"What can I do to protect myself and prevent the spread of disease?","Answer":"Stay home if you feel unwell. If you have a fever, cough and difficulty breathing, seek medical attention and call in advance. Follow the directions of your local health authority."},
        {"Context":"What can I do to protect myself and prevent the spread of disease?","Answer":"If possible, avoid traveling to places �� especially if you are an older person or have diabetes, heart or lung disease."}
        ,
        {"Context":"What can I do to protect myself and prevent the spread of disease?","Answer":"Keep up to date on the latest COVID-19 hotspots (cities or local areas where COVID-19 is spreading widely)."},
        {"Context":"What can I do to protect myself andProtection measures for persons who are in or have recently visited (past 14 days) areas where COVID-19 is spreading prevent the spread of disease?","Answer":"Self-isolate by staying at home if you begin to feel unwell, even with mild symptoms such as headache, low grade fever (37.3 C or above) and slight runny nose, until you recover."},
        {"Context":"Protection measures for persons who are in or have recently visited (past 14 days) areas where COVID-19 is spreading","Answer":"If you develop fever, cough and difficulty breathing, seek medical advice promptly as this may be due to a respiratory infection or other serious condition. Call in advance and tell your provider of any recent travel or contact with travelers."},
        {"Context":"How likely am I to catch COVID-19?","Answer":"The risk depends on where you �are - and more specifically, whether there is a COVID-19 outbreak unfolding there."}
        ,
        {"Context":"How likely am I to catch COVID-19?","Answer":"For most people in most locations the risk of catching COVID-19 is still low. However, there are now places around the world (cities or areas) where the disease is spreading. For people living in, or visiting, these areas the risk of catching COVID-19 is higher."},
        {"Context":"How likely am I to catch COVID-19?","Answer":"COVID-19 outbreaks can be contained and transmission stopped, as has been shown in China and some other countries. Unfortunately, new outbreaks can emerge rapidly. It�s important to be aware of the situation where you are or intend to go."}
        ,
        {"Context":"Should I worry about COVID-19","Answer":"Illness due to COVID-19 infection is generally mild, especially for children and young adults. However, it can cause serious illness: about 1 in every 5 people who catch it need hospital care."},
        {"Context":"Should I worry about COVID-19","Answer":"We can channel our concerns into actions to protect ourselves, our loved ones and our communities."},
        {"Context":"Who is at risk of developing severe illness?","Answer":"While we are still learning about how COVID-2019 affects people, older persons and persons with pre-existing medical conditions (such as high blood pressure, heart disease, lung disease, cancer or diabetes)  appear to develop serious illness more often than others."},
        {"Context":"Are antibiotics effective in preventing or treating the COVID-19?","Answer":"No. Antibiotics do not work against viruses, they only work on bacterial infections. COVID-19 is caused by a virus, so antibiotics do not work.�"}
        ,
        {"Context":"Are antibiotics effective in preventing or treating the COVID-19?","Answer":"Antibiotics should not be used as a means of prevention or treatment of COVID-19. They should only be used as directed by a physician to treat a bacterial infection.�"}
        ,
        {"Context":"Are there any medicines or therapies that can prevent or cure COVID-19?","Answer":"While some western, traditional or home remedies may provide comfort and alleviate symptoms of COVID-19, there is no evidence that current medicine can prevent or cure the disease.�"}
        ,
        {"Context":"Are there any medicines or therapies that can prevent or cure COVID-19?","Answer":"WHO does not recommend self-medication with any medicines, including antibiotics, as a prevention or cure for COVID-19."},
        {"Context":"Is there a vaccine, drug or treatment for COVID-19?","Answer":"Not yet. To date, there is no vaccine and no specific antiviral medicine to prevent or treat COVID-2019.�"}
        ,
        {"Context":"Is there a vaccine, drug or treatment for COVID-19?","Answer":"Those affected should receive care to relieve symptoms. People with serious illness should be hospitalized. Most patients recover thanks to supportive care."},
        {"Context":"Is there a vaccine, drug or treatment for COVID-19?","Answer":"Possible vaccines and some specific drug treatments are under investigation. They are being tested through clinical trials. WHO is coordinating efforts to develop vaccines and medicines to prevent and treat COVID-19."},
        {"Context":"Is there a vaccine, drug or treatment for COVID-19?","Answer":"The most effective ways to protect yourself and others against COVID-19 are to frequently clean your hands, cover your cough with the bend of elbow or tissue, and maintain a distance of at least 1 meter (3 feet) from people who are coughing or sneezing."},
        {"Context":"Is COVID-19 the same as SARS?","Answer":"No. The virus that causes COVID-19 and the one that caused the outbreak of Severe Acute Respiratory Syndrome (SARS) in 2003 are related to each other genetically, but the diseases they cause are quite different."},
        {"Context":"Is COVID-19 the same as SARS?","Answer":"SARS was more deadly but much less infectious than COVID-19. There have been no outbreaks of SARS anywhere in the world since 2003."},
        {"Context":"Should I wear a mask to protect myself","Answer":"Only wear a mask if you are ill with COVID-19 symptoms (especially coughing) or looking after someone who may have COVID-19. Disposable face mask can only be used once."},
        {"Context":"Should I wear a mask to protect myself","Answer":"If you are not ill or looking after someone who is ill then you are wasting a mask.�"}
        ,
        {"Context":"Should I wear a mask to protect myself","Answer":"There is a world-wide shortage of masks, so WHO urges people to use masks wisely."},
        {"Context":"How to put on, use, take off and dispose of a mask?","Answer":"Remember, a mask should only be used by health workers, care takers, and individuals with respiratory symptoms, such as fever and cough."},
        {"Context":"How to put on, use, take off and dispose of a mask?","Answer":"Before touching the mask, clean hands with an alcohol-based hand rub or soap and water"},
        {"Context":"How long is the incubation period for COVID-19?","Answer":"The �incubation period� means the time between catching the virus and beginning to have symptoms of the disease."}
        ,
        {"Context":"How long is the incubation period for COVID-19?","Answer":"Most estimates of the incubation period for COVID-19 range from 1-14 days, most commonly around five days. These estimates will be updated as more data become available."},
        {"Context":"Can humans be infected with the COVID-19 from an animal source?","Answer":"Coronaviruses are a large family of viruses that are common in animals. Occasionally, people get infected with these viruses which may then spread to other people."},
        {"Context":"Can humans be infected with the COVID-19 from an animal source?","Answer":"SARS-CoV was associated with civet cats and MERS-CoV is transmitted by dromedary camels."},
        {"Context":"Can humans be infected with the COVID-19 from an animal source?","Answer":"Possible animal sources of COVID-19 have not yet been confirmed. �"}
        ,
        {"Context":"Can humans be infected with the COVID-19 from an animal source?","Answer":"To protect yourself, such as when visiting live animal markets, avoid direct contact with animals and surfaces in contact with animals."},
        {"Context":"Can humans be infected with the COVID-19 from an animal source?","Answer":"Handle raw meat, milk or animal organs with care to avoid contamination of uncooked foods and avoid consuming raw or undercooked animal products."},
        {"Context":"Can I catch COVID-19 from my pet?","Answer":"While there has been�one instance of a dog being infected in Hong Kong, to�date, there is no evidence that a dog, cat or any pet can transmit COVID-19."}
        ,
        {"Context":"Can I catch COVID-19 from my pet?","Answer":"COVID-19 is mainly spread through droplets produced when an infected person coughs, sneezes, or speaks."},
        {"Context":"How long does the virus survive on surfaces?","Answer":"It is not certain how long the virus that causes COVID-19 survives on surfaces, but it seems to behave like other coronaviruses."},
        {"Context":"How long does the virus survive on surfaces?","Answer":"Studies suggest that coronaviruses (including preliminary information on the COVID-19 virus) may persist on surfaces for a few hours or up to several days. This may vary under different conditions (e.g. type of surface, temperature or humidity of the environment)."},
        {"Context":"How long does the virus survive on surfaces?","Answer":"If you think a surface may be infected, clean it with simple disinfectant to kill the virus and protect yourself and others."},
        {"Context":"Is it safe to receive a package from any area where COVID-19 has been reported?","Answer":"Yes. The likelihood of an infected person contaminating commercial goods is low and the risk of catching the virus that causes COVID-19 from a package that has been moved, travelled, and exposed to different conditions and temperature is also low.�"}
        ,
        {"Context":"Is there anything I should not do?","Answer":"Smoking"},
        {"Context":"Is there anything I should not do?","Answer":"Wearing multiple masks"},
        {"Context":"Is there anything I should not do?","Answer":"Taking antibiotics"},
        {"Context":"Is the source causing COVID-19 known?","Answer":"Currently, the source of SARS-CoV-2, the coronavirus (CoV) causing COVID-19 is unknown."},
        {"Context":"How did the first human SARS-CoV-2 infections occur?","Answer":"The first human cases of COVID-19 were identified in Wuhan City, China in December 2019."},
        {"Context":"How did the first human SARS-CoV-2 infections occur?","Answer":"SARS-CoV, the virus which caused the SARS outbreak in 2003, jumped from an animal reservoir (civet cats, a farmed wild animal) to humans and then spread between humans.�"}
        ,
        {"Context":"How did the first human SARS-CoV-2 infections occur?","Answer":"Until the source of this virus is identified and controlled, there is a risk of reintroduction of the virus in the human population and the risk of new outbreaks like the ones we are currently experiencing."},
        {"Context":"Is COVID-19 airborne?","Answer":"The virus that causes COVID-19 is mainly transmitted through droplets generated when an infected person coughs, sneezes, or speaks. These droplets are too heavy to hang in the air. They quickly fall on floors or surfaces."},
        {"Context":"Are pregnant women at higher risk from COVID-19?","Answer":"At present there is no evidence that they are at higher risk of severe illness than the general population."},
        {"Context":"Are pregnant women at higher risk from COVID-19?","Answer":"due to changes in their bodies and immune systems, we know that pregnant women can be badly affected by some respiratory infections. It is therefore important that they take precautions to protect themselves against COVID-19, and report possible symptoms (including fever, cough or difficulty breathing) to their healthcare provider."},
        {"Context":"I'm pregnant. How can I protect myself against COVID-19?","Answer":"Pregnant women should take the same precautions to avoid COVID-19 infection as other people."},
        {"Context":"I'm pregnant. How can I protect myself against COVID-19?","Answer":"Pregnant women and women who have recently delivered � including those affected by COVID-19 - should attend their routine care appointments.�"}
        ,
        {"Context":"Should pregnant women be tested for COVID-19?","Answer":"Testing protocols and eligibility vary depending on where you live."},
        {"Context":"Can COVID-19 be passed from a woman to her unborn baby?","Answer":"We still do not know if a pregnant woman with COVID-19 can pass the virus to her foetus or baby during pregnancy or delivery. To date, the virus has not been found in samples of amniotic fluid or breastmilk."},
        {"Context":"What care should be available during pregnancy and childbirth?","Answer":"All pregnant women, including those with confirmed or suspected COVID-19 infections, have the right to high quality care before, during and after childbirth.�"}
        ,
        {"Context":"Do pregnant women with suspected or confirmed COVID-19 need to give birth by ceasarean section?","Answer":"No. WHO advice is that caesarean sections should only be performed when medically justified."},
        {"Context":"Can women with COVID-19 breastfeed?","Answer":"Yes. Women with COVID-19 can breastfeed if they wish to do so."},
        {"Context":"Can I touch and hold my newborn baby if I have COVID-19?","Answer":"Yes. Close contact and early, exclusive breastfeeding helps a baby to thrive."},
        {"Context":"How are COVID-19 and influenza viruses similar?","Answer":"COVID-19 and influenza viruses have a similar disease presentation. That is, they both cause respiratory disease, which presents as a wide range of illness from asymptomatic or mild through to severe disease and death."},
        {"Context":"How are COVID-19 and influenza viruses similar?","Answer":"Both viruses are transmitted by contact, droplets and fomites."},
        {"Context":"How are COVID-19 and influenza viruses different?","Answer":"The speed of transmission is an important point of difference between the two viruses. Influenza has a shorter median incubation period (the time from infection to appearance of symptoms) and a shorter serial interval (the time between successive cases) than COVID-19 virus."},
        {"Context":"How are COVID-19 and influenza viruses different?","Answer":"Those most at risk for severe influenza infection are children, pregnant women, elderly, those with underlying chronic medical conditions and those who are immunosuppressed."},
        {"Context":"How are COVID-19 and influenza viruses different?","Answer":"Mortality for COVID-19 appears higher than for influenza, especially seasonal influenza."},
        {"Context":"What medical interventions are available for COVID-19 and influenza viruses?","Answer":"While there are a number of therapeutics currently in clinical trials in China and more than 20 vaccines in development for COVID-19, there are currently no licensed vaccines or therapeutics for COVID-19."},
        {"Context":"Are smokers and tobacco users at higher risk of COVID-19 infection?","Answer":"Smokers are likely to be more vulnerable to COVID-19 as the act of smoking means that fingers (and possibly contaminated cigarettes) are in contact with lips which increases the possibility of transmission of virus from hand to mouth."},
        {"Context":"Are smokers and tobacco users at higher risk of COVID-19 infection?","Answer":"Smoking products such as water pipes often involve the sharing of mouth pieces and hoses, which could facilitate the transmission of COVID-19 in communal and social settings."},
        {"Context":"How large does a meeting or event need to be in order to be a �mass gathering�?","Answer":"High profile international sporting events such as the Olympics or World Cups as well as international religious events such as the Hajj count as mass gatherings."}
        ,
        {"Context":"How large does a meeting or event need to be in order to be a �mass gathering�?","Answer":"An event counts as a �mass gatherings� if the number of people it brings together is so large that it has the potential to strain the planning and response resources of the health system in the community where it takes place.�"}
        ,
        {"Context":"Does WHO recommend that all international mass gatherings be cancelled because of COVID-19?","Answer":"No. As each international mass gathering is different, the factors to consider when determining if the event should be cancelled may also differ.�"}]
          
          var questions = []
          var answers = []
          for (row in data)
          {
              answers.push(data[row]['Answer'])
              questions.push(data[row]['Context'])
          }
          
          const test = [
            query
          ];
          model.embed(test).then(embeddings => {
            const question_encodings = embeddings.arraySync();
            model.embed(answers).then(embeddings => {
            const response_encodings = embeddings.arraySync();
            var scores = []
            for (let i = 0; i < question_encodings.length; i++) {
            for (let j = 0; j < response_encodings.length; j++) {
              scores.push(dotProduct(question_encodings[i], response_encodings[j]));
            }
          };
          const arg_max = scores.indexOf(Math.max(...scores));
          const result = answers[arg_max]
          return result
          });
          });
          });
          }