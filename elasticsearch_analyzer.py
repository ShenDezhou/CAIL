import elasticsearch

print('BEGIN')

es = elasticsearch.Elasticsearch('http://192.168.4.250:9200/', timeout=30,)
es.cluster.health()
search_args = {}
search_args['index']='lbzlaw'
search_args['doc_type']='lbzlaw'
search_args['size']=10
search_args['body']='{}'

res = es.search(**search_args)
print(res)

search_args['body']='''
{
  "text": "测试elasticsearch分词器",
  "analyzer": "ik_smart"
}
'''
search_args.pop('size')
search_args.pop('doc_type')
res = es.indices.analyze(**search_args)
print(res)
print('FIN')