import os
import time
import argparse
import networkx as nx
from nodevectors import Node2Vec


def main(args):
    print(args)
    edgelists = [qf for qf in os.listdir(args.input)
                 if qf.endswith('.edgelist') and not qf.startswith('_')]
    g = None

    exclude = args.exclude or []

    print('loading edgelists...')
    for eg in edgelists:
        if eg in exclude or eg.rsplit('.', 1)[0] in exclude:
            continue
        print('- ' + eg)
        h = nx.read_edgelist(os.path.join(args.input, eg), nodetype=str, create_using=nx.DiGraph(), delimiter=' ')
        for edge in h.edges():
            h[edge[0]][edge[1]]['weight'] = 1

        g = h if g is None else nx.compose(g, h)

    g = g.to_undirected()

    print('Nodes: %d' % nx.number_of_nodes(g))
    print('Edges: %d' % nx.number_of_edges(g))

    print('Start learning at %s' % time.asctime())
    g2v = Node2Vec(
        walklen=args.walk_length,
        epochs=args.num_walks,
        n_components=args.dimensions,
        return_weight=1 / args.p,
        neighbor_weight=1 / args.q,
        threads=args.workers,
        w2vparams={
            'window': args.window_size,
            'epochs': args.iter,  # TODO: Was iter; changed to epochs by gensim
            'batch_words': 128,
            'min_count': 0,
            'negative': 25,
            'sg': 1
        },
        verbose=True
    )
    g2v.fit(g)
    print('End learning at %s' % time.asctime())

    # Save model to gensim.KeyedVector format
    g2v.save_vectors(args.output)


def parse_args():
    parser = argparse.ArgumentParser(description="Run edgelist 2 vec.")

    parser.add_argument('-i', '--input', default='./edgelist',
                        help='Input graph (edgelists) path')

    parser.add_argument('-o', '--output', default='embeddings.bin',
                        help='Output file name')

    parser.add_argument('--walk_length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num_walks', type=int, default=40,
                        help='Number of walks per source. Default is 40.')

    parser.add_argument('-p', type=float, default=1,
                        help='Return hyper-parameter. Default is 1.')

    parser.add_argument('-q', type=float, default=1,
                        help='Inout hyper-parameter. Default is 1.')

    parser.add_argument('--dimensions', type=int, default=100,
                        help='Number of dimensions. Default is 100.')

    parser.add_argument('--window-size', type=int, default=5,
                         help='Context size for optimization. Default is 5.')

    parser.add_argument('--iter', type=int, default=5,
                        help='Number of epochs in word2vec')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0 (full use).')

    parser.add_argument('--exclude', nargs='+',
                        help='Edgelists to exclude')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
