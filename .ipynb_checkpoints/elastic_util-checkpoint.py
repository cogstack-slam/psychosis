import json
from datetime import date, datetime
import numpy as np

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))
    
def rec_to_actions(df, INDEX, TYPE, doc_index):
    for record in df.to_dict(orient="records"):
        yield ('{ "index" : { "_index" : "%s", "_type" : "%s",  "_id" : "%s"}}'% (INDEX, TYPE, record[doc_index]))
        yield (json.dumps(record, default=json_serial))
            
def bulk_insert(client, df, INDEX, TYPE, doc_index):
    if not client.indices.exists(INDEX):
    #     raise RuntimeError('index %s does not exists'%INDEX)
        client.indices.create(INDEX)
    n = int(df.shape[0]/3000) + 1
    for d in np.array_split(df, n):
        r = client.bulk(rec_to_actions(d, INDEX, TYPE, doc_index)) # return a dict
        if(r["errors"]):
            return False
    return True

        
def update_to_actions(df, INDEX, TYPE, doc_index):
    for record in df.to_dict(orient="records"):
        yield ('{ "update" : { "_index" : "%s", "_type" : "%s",  "_id" : "%s"}}'% (INDEX, TYPE, record[doc_index]))
        yield ('{ "doc":' + json.dumps(record, default=json_serial) + '}')
        
def bulk_update(client, df, INDEX, TYPE, doc_index):
    if not client.indices.exists(INDEX):
        raise RuntimeError('index %s does not exists'%INDEX)
#         client.indices.create(INDEX)
    n = int(df.shape[0]/3000) + 1
    for d in np.array_split(df, n):
        r = client.bulk(update_to_actions(d, INDEX, TYPE, doc_index)) # return a dict
        if(r["errors"]):
            return False
    return True