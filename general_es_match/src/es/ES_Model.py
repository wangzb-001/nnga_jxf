# pip install elasticsearch
from elasticsearch import Elasticsearch
from tqdm import tqdm
from elasticsearch import helpers
from config import Config


class ES_Model:
    '''
    基于Elasticsearch（BM25)的文本匹配模型
    '''
    es_client = Elasticsearch(
        [
            {'host': Config.config['es_host'],
             'port': Config.config['es_port'],
             "timeout": 10}])

    @classmethod
    def get_indexs(cls):
        indexs = cls.es_client.indices.get_alias().keys()
        return [i for i in indexs if 'model_' in i]

    @classmethod
    def inser_docs(cls, index, docs):
        if cls.es_client.indices.exists(index):
            cls.index_delete(index)
            cls.index_create(index)
        actions = []
        for i, doc in enumerate(tqdm(docs)):
            # body = {'text': doc}
            id = str(index) + '_' + str(i)
            action = {
                "_index": index,
                "_id": id,
                "_source": {
                    "text": doc}
            }
            actions.append(action)
        helpers.bulk(cls.es_client, actions=actions)

    @classmethod
    def clear(cls):
        # 删除所有数据
        indexs = cls.get_indexs()
        for index in indexs:
            cls.index_delete(index)

    @classmethod
    def index_create(cls, index):
        if cls.es_client.indices.exists(index):
            cls.index_delete(index)
        # 索引：类似分区的概念，指定index后只会对index匹配区的数据进行搜索
        return cls.es_client.indices.create(index=index)

    @classmethod
    def index_get(cls, index, id=None):
        if cls.es_client.indices.exists(index):
            if id is None:
                return cls.es_client.indices.get(index=index)
            else:
                return cls.es_client.indices.get(index=index, id=id)

    @classmethod
    def index_delete(cls, index):
        return cls.es_client.indices.delete(index)

    @classmethod
    def delete_by_id(cls, index, id):
        # 根据_id删除
        return cls.es_client.delete(index=index, id=id)

    @classmethod
    def delete_by_query(cls, index, body):
        # 查询条件参数
        cls.es_client.delete_by_query(index=index, body=body)

    @classmethod
    def insert_doc(cls, index, doc, id):
        body = {'text': doc}
        id = str(index) + '_' + str(id)
        cls.es_client.index(index=index, doc_type=str, id=id, body=body)

    @classmethod
    def query_doc(cls, doc, size=10, index=None):
        body = {'query': {'match': {'text': doc}}, 'size': size}
        return cls.es_client.search(body=body, index=index)

    @classmethod
    def query_docs(cls, docs, size=10, index=None):
        # 批量查询
        request = []
        for doc in docs:
            req_head = {'index': index}
            req_body = {'query': {'match': {'text': doc}}, 'size': size}
            request.extend([req_head, req_body])

        res = cls.es_client.msearch(body=request, index=index)
        return res
