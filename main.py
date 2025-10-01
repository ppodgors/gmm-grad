# main.py
import argparse
from src.experiments.olivetti.run import run_olivetti_experiment
from src.experiments.newsgroups.run import run_newsgroups_experiment
from src.experiments.newsgroups import config as newsgroups_config
from src.experiments.synthetic.run import run_synthetic_experiment
from src.experiments.synthetic.plotting import run_plotting

def main():
    parser = argparse.ArgumentParser(description="Main script to run experiments.")
    subparsers = parser.add_subparsers(dest='experiment', required=True, help='The experiment to run')
    # --- Synthetic Experiment Parser ---
    parser_synth = subparsers.add_parser('synthetic', help='Run the synthetic data experiment.')
    synth_subparsers = parser_synth.add_subparsers(dest='action', required=True, help='Action to perform: run or plot')
    parser_synth_run = synth_subparsers.add_parser('run', help='Run the simulation.')
    parser_synth_run.add_argument('--n_samples', type=int, required=True, help="Number of samples per component.")
    parser_synth_run.add_argument('--num_worlds', type=int, default=25, help="Number of worlds to generate.")
    parser_synth_run.add_argument('--num_reps', type=int, default=4, help="Number of replications per world.")
    parser_synth_run.add_argument('--d_start', type=int, default=20, help="Starting number of features (d).")
    parser_synth_run.add_argument('--d_stop', type=int, default=101, help="Ending number of features (d), exclusive.")
    parser_synth_run.add_argument('--d_step', type=int, default=10, help="Step for the number of features range.")
    
    # Sub-parser for the 'plot' action
    parser_synth_plot = synth_subparsers.add_parser('plot', help='Plot results from a file.')
    parser_synth_plot.add_argument('--input_file', type=str, required=True, help="Path to the .npz results file.")

    
    # --- Olivetti Parser ---
    parser_olivetti = subparsers.add_parser('olivetti', help='Run the Olivetti Faces experiment.')
    parser_olivetti.add_argument('--n_rep', type=int, default=25, help='Number of repetitions.')
    parser_olivetti.add_argument('--main_seed', type=int, default=27, 
                        help='The main random seed for initialization.')
    parser_olivetti.add_argument('--subjects', type=int, default=10, 
                        help='Number of subjects to include from the dataset.')
    parser_olivetti.add_argument('--output_dir', type=str, default='Olivetti_results', 
                        help='Directory to save the output results.')

    # --- 20 Newsgroups Parser ---
    parser_newsgroups = subparsers.add_parser('newsgroups', help='Run the 20 Newsgroups experiment.')
    parser_newsgroups.add_argument('--worlds', type=int, default=newsgroups_config.N_WORLDS,
                                   help='Number of data worlds to generate and process.')
    parser_newsgroups.add_argument('--reps', type=int, default=newsgroups_config.N_REPS,
                                   help='Number of repetitions per world.')
    parser_newsgroups.add_argument('--samples_per_cat', type=int, default=newsgroups_config.SAMPLES_PER_CAT,
                                   help='Number of documents to sample per category.')
    parser_newsgroups.add_argument('--data_dir', type=str, default=newsgroups_config.DATA_DIR,
                                   help='Directory to store/read data files.')
    parser_newsgroups.add_argument('--categories', nargs='+', default=newsgroups_config.CATEGORIES,
                                   help='List of newsgroup categories to use (space-separated).')

    args = parser.parse_args()

    if args.experiment == 'synthetic':
        if args.action == 'run':
            run_synthetic_experiment(args)
        elif args.action == 'plot':
            run_plotting(args)
    elif args.experiment == 'olivetti':
        print(f"Executing experiment: 'Olivetti'...")
        run_olivetti_experiment(
            n_rep=args.n_rep,
            main_seed=args.main_seed,
            num_subjects=args.subjects,
            output_dir=args.output_dir
        )
    elif args.experiment == 'newsgroups':
        print(f"Executing experiment: '20 Newsgroups'...")
        run_newsgroups_experiment(
            n_worlds=args.worlds,
            n_reps=args.reps,
            data_dir=args.data_dir,
            categories=args.categories,
            samples_per_cat=args.samples_per_cat
        )

if __name__ == "__main__":
    main()