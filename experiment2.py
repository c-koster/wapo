d = {"id": "fcab630ae35becc789078150e1628136", "ner": "Amtrak/ORG\tNortheast Corridor/NORP\tFriday/DATE\tfour days/DATE\tNew York\u2019s/GPE\tPennsylvania/GPE\tAmtrak\u2019s/ORG\tThursday/DATE\tAmtrak/ORG\tWick Moorman/PERSON\tfour days/DATE\tMonday/DATE\tmorning/TIME\tPenn Station/ORG\tAcela/PERSON\tMarch 24/DATE\tAmtrak/ORG\tNew York Penn Station/ORG\tMoorman/PERSON\tNortheast Corridor/NORP\tAmtrak/ORG\tMonday/DATE\tjust one day/DATE\tthe New York Times/ORG\tNew Jersey/GPE\tChris Christie/PERSON\tAmtrak/ORG\tAmtrak/ORG\tbetween $2.5 million and $5 million/MONEY\tthe Hudson River Tunnels/LOC\tNortheast Corridor/NORP\tTimes/ORG\tMoorman/PERSON\tMarch 24/DATE\ttwo/CARDINAL\tAcela/PERSON\tMonday/DATE\tjust after 9 a.m./TIME\tMoorman/PERSON\tMoorman/PERSON\tAmtrak/ORG\tthe Federal Railroad Administration/ORG\tFRA/ORG\tAmtrak/ORG\tPenn Station/ORG\tMonday/DATE\tMoorman/PERSON\tPenn Station/ORG"}
d1 = {"id": "cdda2a7f8128b81ad6956aa8bdba0df5", "ner": "RNC/ORG\tMichael Steele/PERSON\tTrump/ORG\tCorey Lewandowski/PERSON\tJoe\u2019s Seafood/ORG\tWednesday/DATE\tthe White House/FAC\tSteele/PERSON\tLewandowski\u2019s old/ORG\ttwo/CARDINAL\tLewandowski/PERSON\tOne America News Network/ORG\tSteele/PERSON\tMSNBC/ORG"}


#print(d.keys())

# ok

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

ner_list = d1['ner'].split('\t')
print([i.split('/')[0] for i in ner_list if stoplabels[i.split('/')[1]]])
