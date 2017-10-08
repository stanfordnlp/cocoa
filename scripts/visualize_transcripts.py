from argparse import ArgumentParser

from cocoa.core.util import write_json

from analysis.visualizer import Visualizer
from analysis.html_visualizer import HTMLVisualizer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--survey-transcripts', nargs='+',
            help='Path to directory containing evaluation transcripts')
    parser.add_argument('--dialogue-transcripts', nargs='+',
            help='Path to directory containing dialogue transcripts')
    parser.add_argument('--worker-ids', nargs='+',
            help='Path to json file containing chat_id to worker_id mappings')
    parser.add_argument('--summary', default=False, action='store_true',
            help='Summarize human ratings')
    parser.add_argument('--hist', default=False, action='store_true',
            help='Plot histgram of ratings')
    parser.add_argument('--html-visualize', action='store_true',
            help='Output html files')
    parser.add_argument('--outdir', default='.', help='Output dir')
    parser.add_argument('--stats', default='stats.json',
            help='Path to stats file')
    parser.add_argument('--partner', default=False, action='store_true',
            help='Whether this is from partner survey')
    HTMLVisualizer.add_html_visualizer_arguments(parser)
    args = parser.parse_args()

    visualizer = Visualizer(args.dialogue_transcripts,
            args.survey_transcripts, args.worker_ids)

    visualizer.compute_effectiveness()

    # TODO: move summary and hist to analyzer
    if args.hist:
        visualizer.hist(question_scores, args.outdir, partner=args.partner)

    if args.summary:
        summary = visualizer.summarize()
        write_json(summary, args.stats)

    if args.worker_ids:
        visualizer.worker_stats()

    if args.html_output:
        visualizer.html_visualize(args.viewer_mode, args.html_output,
            css_file=args.css_file, img_path=args.img_path,
            worker_ids=visualizer.worker_ids)
