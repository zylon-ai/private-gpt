import argparse


class ArgsHandler:
    PRIVATE_GPT = 'privategpt'
    INGEST = 'ingest'

    @staticmethod
    def parse_args(description, arguments):
        parser = argparse.ArgumentParser(description=description)
        for arg in arguments:
            parser.add_argument(*arg['args'], **arg['kwargs'])
        return parser.parse_args()

    @staticmethod
    def get_args(type):
        if type == ArgsHandler.PRIVATE_GPT:
            description = 'privateGPT: Ask questions to your documents without an internet connection, using the ' \
                          'power of LLMs.'
            arguments = [
                {
                    'args': ["--hide-source", "-S"],
                    'kwargs': {'action': 'store_true',
                               'help': 'Use this flag to disable printing of source documents used for answers.'}
                },
                {
                    'args': ["--mute-stream", "-M"],
                    'kwargs': {'action': 'store_true',
                               'help': 'Use this flag to disable the streaming StdOut callback for LLMs.'}
                },
            ]
        elif type == ArgsHandler.INGEST:
            description = 'Ingest: Document processing script to parse the document and create embeddings locally'
            arguments = [
                {
                    'args': ['--persist_dir'],
                    'kwargs': {'metavar': 'DIRECTORY_PATH', 'type': str, 'help': 'Directory to persist data'}
                },
                {
                    'args': ['--source_dir'],
                    'kwargs': {'metavar': 'DIRECTORY_PATH', 'type': str, 'help': 'Directory for source documents'}
                },
                {
                    'args': ['--embeddings_model'],
                    'kwargs': {'metavar': 'MODEL_NAME', 'type': str, 'help': 'Name of the embeddings model'}
                },
            ]
        else:
            raise ValueError(f"Unknown type: {type}")

        return ArgsHandler.parse_args(description, arguments)
