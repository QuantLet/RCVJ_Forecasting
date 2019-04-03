import requests
import time
import pymongo
import datetime as dt

block_request_raw = 'https://blockchain.info/rawblock/'
latest_block_hash = '0000000000000000002427f67f85a3116048c061a16203072353343af7894ca5'


class BlockInfoCollector:

    def __init__(self, start_block_hash):
        self.blocks_chain = list()  # save block chain in a list
        self.block_current = dict()
        self.block_counter = 0
        self.prev_block_hash = start_block_hash

    def block_visiting(self, block_hash):
        request_get = requests.get(block_request_raw + block_hash)
        if request_get.status_code == 200:
            block_info = request_get.json()
            self.block_current = block_info
            print(dt.datetime.fromtimestamp(self.block_current['time']))
            return block_info
        else:
            raise IOError("**********Error! Connection Error!**********")

    def append_blocks(self):
        """
        append current block into block chain
        change previous block hash address
        :return:
        """
        self.blocks_chain.append(self.block_current)
        self.prev_block_hash = self.block_current['prev_block']
        self.block_counter += 1

    def block_roller(self):

        if self.block_counter == 0:
            # Initial the block
            latest_block_height = self.read_block()  # start with reading the database and get the latest block's prev block hash
            print(f'The latest block height is: {latest_block_height}')
            self.block_visiting(self.prev_block_hash)  # Visit the previous block
            self.append_blocks()  # Append the previous block and get ready
            print(f'Block # {self.block_current["height"]} initialed')

        # time.sleep(10)

        _height = self.block_current['height']
        while _height > 0:
            self.block_visiting(self.prev_block_hash)
            self.append_blocks()
            print(self.block_counter)
            _height = self.block_current['height']
            print(f'Block # {_height} is finished')
            self.write_block()
            # time.sleep(10)

    def write_block(self):
        client = pymongo.MongoClient('localhost', 27017)
        db_btc_blochain = client['Blockchain']['BTC_Blockchain']
        try:
            db_btc_blochain.insert_one(self.block_current)
        except Exception as e:
            print(e)

    def read_block(self):
        client = pymongo.MongoClient('localhost', 27017)
        coll_btc_blochain = client['Blockchain']['BTC_Blockchain']
        heights_cursor = coll_btc_blochain.find({}, {'height': True, '_id': False})
        heights = [item['height'] for item in heights_cursor]
        lowest_heights = min(heights)
        latest_block = coll_btc_blochain.find_one({'height': lowest_heights},
                                                  {'hash': True, 'prev_block': True, '_id': False})
        self.prev_block_hash = latest_block['prev_block']
        return lowest_heights

    def parse_block(self, block_info):
        pass


if __name__ == '__main__':
    while True:
        block = BlockInfoCollector(start_block_hash=latest_block_hash)
        try:
            block_info = block.block_roller()
        except Exception as e:
            print(e)
            time.sleep(10)
