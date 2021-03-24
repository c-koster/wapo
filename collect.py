import gzip, json

keep = set()
with open('queries/all.ids', 'r') as fp:
    for line in fp:
        keep.add(line.strip())

found = 0
with open('trec.wapo.mini.jsonl', 'w') as out:
    with gzip.open('TREC_Washington_Post_collection.v3.jl.gz', 'rt') as fp:
        for line in fp:
            doc = json.loads(line)
            if doc['id'] in keep:
                out.write(line)
                found += 1
                print(found)
                # stop rule for now
                if found > 100:
                    exit(0)
