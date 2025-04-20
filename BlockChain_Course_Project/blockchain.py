import hashlib
import time

class Block:
    def __init__(self, index, node_id, accuracy, previous_hash):
        self.index = index
        self.timestamp = time.time()
        self.node_id = node_id
        self.accuracy = accuracy
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = f"{self.index}{self.timestamp}{self.node_id}{self.accuracy}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, "genesis", 0.0, "0")
        self.chain.append(genesis_block)

    def add_block(self, node_id, accuracy):
        prev = self.chain[-1]
        new_block = Block(len(self.chain), node_id, accuracy, prev.hash)
        self.chain.append(new_block)

    def print_chain(self):
        for block in self.chain:
            print(f"Block {block.index} | Node: {block.node_id} | Acc: {block.accuracy:.4f} | Hash: {block.hash[:10]}...")
