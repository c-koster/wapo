# using ~ spacy ~
# let's make a function which takes a text block and returns a set of unique names
import spacy
from spacy import displacy
#import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

stoplabels = {
    "PERSON": True,
    "NORP": True,
    "FAC": True,
    "ORG": True,
    "GPE": True,
    "LOC": True,
    "PRODUCT": True,
    "EVENT": True,
    "WORK_OF_ART": True,
    "LAW": True,
    "LANGUAGE": True,
    "DATE": True,
    "TIME": False,
    "PERCENT": False,
    "MONEY": False,
    "QUANTITY": False,
    "ORDINAL": False,
    "CARDINAL": False,
}

#text = """On Monday, President Trump imposed new sanctions on Iran that targeted the country’s supreme leader, Ayatollah Ali Khamenei, and related entities. It’s just the latest punitive measure aimed at Tehran in the long history of tense relations between the United States and the Islamic republic. But the sanctions come at a moment when Trump says he’s eager to negotiate with the Iranian regime. He signaled as much in recent days. “We’re not going to have Iran have a nuclear weapon ... when they agree to that, they’re going to have a wealthy country,” Trump told reporters outside the White House on Friday. “They’re going to be so happy, and I’m going to be their best friend. I hope that happens.” Those comments, which followed Trump’s decision last week to cancel a retaliatory strike against Iran for shooting down a U.S. drone, have reopened the possibility of negotiations between Washington and Tehran. But this window to talk to Tehran is unlikely to remain open for long. Now is the moment, if there ever was one, for Trump to use what he touts as his creative dealmaking skills to support the Iranian people, while still applying pressure on the regime and its elites. There are options available to him if he truly wants, as he said last week, to “make Iran great again.” The 2015 deal designed to curb Iran’s nuclear activity addressed what the United States and other world powers identified as the biggest challenge to security and peace posed by Iran. Iran was unwilling to engage on other issues, such as its missile program, human rights and support for proxies. By assigning such disproportionate urgency to a nuclear program that our own intelligence agencies believed was not moving toward weaponization, we squandered much of the leverage to address other critical issues. Leaving that deal last year made engagement on any matter even more difficult. Now that Trump — inadvertently or by design — has created an opportunity for fresh talks, he should look for other issues beyond the nuclear program to kick-start new dialogue. Tehran, however, says it’s unwilling to engage in the face of pressure and disrespect. The new measures against Khamenei are likely to harden that position even further, something the architects of the current Iran policy know well. So maybe it’s time for Trump to start making offers Iran can’t refuse. The most effective way to do this would be to begin making public overtures that everyday Iranians would see as genuinely intended to improve their lives. Easing banking restrictions on average citizens to facilitate the vast Iranian diaspora’s ability to send remittances to loved ones in Iran, lifting the travel ban on Iranians to the United States and making technology available to help combat Iran’s environmental crises are tools we have available to negotiate with Iran that would help people without bolstering the regime. In exchange, Trump could ask for the immediate release of U.S. citizens being held by the Iranian regime and demand a permanent end to the opaque and arbitrary detention of Americans. Supporting and empowering the Iranian people, who continue to be the primary victims of the Trump administration’s Iran policy, must be an imperative. Applying more U.S. sanctions on Iran would only further degrade the fragile Iranian economy and exacerbate the suffering of ordinary Iranians. That is why a growing number of civil society activists inside Iran — the same people we claim to support through sanctions on the regime — are calling for a de-escalation of tensions and fewer sanctions. Opening the way for remittances to be delivered and Iranians to have people-to-people contact with the United States would help make the case that the Trump administration isn’t trying to hurt the Iranian public. Another option would be making technology available to Iran to combat the country’s epic water shortages, choking pollution and woeful earthquake preparedness. Coupling these incentives to directly help Iranians with the release of U.S. hostages would give the administration the opportunity to do two things it has publicly committed to yet failed to deliver on: bring Americans home and support the aspirations of Iranians. Secretary of State Mike Pompeo repeatedly says the United States is ready to engage once Iran starts behaving like a “normal country.” Perhaps if we give Iranian leaders the opportunity to act normally — in increments — they will. This would not provide a direct lifeline to the regime — something hawks in the Trump administration would oppose — but it would ease some of the growing public suffering over the country’s economic plight. If the regime doesn’t bite, it would do much more to sow discontent among Iranians than anything the Trump administration has done so far. Helping the country brace for potential humanitarian crises would create a nonpolitical channel of cooperation but also highlight the U.S. commitment to improving the conditions of people, regardless of where they live in the world. In the long term, it would do more to open a constructive channel for negotiations than any of the Trump administration’s “maximum pressure” policies are likely to achieve. Read more: Jason Rezaian: Iran is outmatched in its latest game of rhetorical chicken. But it might be too late. Ardeshir Zahedi and Ali Vaez: The U.S. should strive for a stable Iran. Instead, it is suffocating it. Dennis Ross: Trump is a belligerent isolationist. But on Iran, he needs allies. Jason Rezaian: What the Iran crisis tells us about Trump’s lack of credibility Max Boot: In Iran crisis, our worst fears about Trump are realized"""

def get_named_entities(text):
    doc = nlp(text)
    # for ent in doc.ents: print(ent.text, ent.label_)

    # look at my shiny for loop thing
    items = [x.text for x in doc.ents if stoplabels[x.label_]]
    return set(items)


# this takes way too long to be practical
