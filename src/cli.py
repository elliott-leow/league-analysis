
import argparse
import sys
import logging
from pathlib import Path

from .config import ALL_RANKS, FIGURES_DIR, REPORTS_DIR, SAMPLE_SIZE
from . import preprocessing, ges, parameters, queries, visualize, compare

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_preprocess(args):
    logger.info("Starting preprocessing")
    
    if args.rank == "all":
        preprocessing.preprocess_all_ranks(sample_size=args.sample_size)
    else:
        if args.rank not in ALL_RANKS:
            logger.error(f"Invalid rank: {args.rank}. Choose from {ALL_RANKS}")
            return
        
        df = preprocessing.preprocess_for_rank(args.rank, sample_size=args.sample_size)
        preprocessing.save_processed_data(df, args.rank)
        
        logger.info(f"Preprocessing complete for {args.rank}")
        logger.info(f"Dataset shape: {df.shape}")


def cmd_learn(args):
    
    # handle "all" ranks
    if args.rank == "all":
        logger.info("Learning structure for all ranks")
        for rank in ALL_RANKS:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing rank: {rank}")
            logger.info(f"{'='*50}")
            # create a copy of args with specific rank
            rank_args = argparse.Namespace(**vars(args))
            rank_args.rank = rank
            cmd_learn(rank_args)
        return
    
    logger.info(f"Learning structure for rank: {args.rank}")
    
    # validate rank
    if args.rank not in ALL_RANKS:
        logger.error(f"Invalid rank: {args.rank}. Choose from {ALL_RANKS}")
        return
    
    # load data
    try:
        data = preprocessing.load_processed_data(args.rank)
    except FileNotFoundError:
        logger.error(f"Processed data not found for {args.rank}. Run preprocessing first.")
        return
    
    # run GES
    result = ges.fit_ges(data, use_constraints=not args.no_constraints)
    
    # save result
    ges.save_ges_result(result, args.rank)
    
    # create visualization
    output_file = FIGURES_DIR / f"cpdag_{args.rank}.png"
    visualize.plot_cpdag(
        result['edges'],
        result['variables'],
        title=f"CPDAG - {args.rank} ({result['n_edges']} edges)",
        output_file=output_file,
        layout=args.layout
    )
    
    # save DOT file
    dot_file = FIGURES_DIR / f"cpdag_{args.rank}.dot"
    visualize.save_graph_as_dot(
        result['edges'],
        result['variables'],
        dot_file,
        title=f"CPDAG_{args.rank}"
    )
    
    logger.info(f"Structure learning complete. Found {result['n_edges']} edges.")
    logger.info(f"Saved visualization to {output_file}")


def cmd_params(args):
    
    # handle "all" ranks
    if args.rank == "all":
        logger.info("Learning parameters for all ranks")
        for rank in ALL_RANKS:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing rank: {rank}")
            logger.info(f"{'='*50}")
            # create a copy of args with specific rank
            rank_args = argparse.Namespace(**vars(args))
            rank_args.rank = rank
            cmd_params(rank_args)
        return
    
    logger.info(f"Learning parameters for rank: {args.rank}")
    
    # validate rank
    if args.rank not in ALL_RANKS:
        logger.error(f"Invalid rank: {args.rank}. Choose from {ALL_RANKS}")
        return
    
    # load data
    try:
        data = preprocessing.load_processed_data(args.rank)
    except FileNotFoundError:
        logger.error(f"Processed data not found for {args.rank}. Run preprocessing first.")
        return
    
    # load GES result
    try:
        ges_result = ges.load_ges_result(args.rank)
    except FileNotFoundError:
        logger.error(f"GES result not found for {args.rank}. Run structure learning first.")
        return
    
    # learn parameters
    model = parameters.learn_parameters_from_ges(ges_result, data)
    
    # save model
    parameters.save_bayesian_network(model, args.rank)
    
    # validate
    validation = parameters.validate_cpts(model)
    if validation["valid"]:
        logger.info("CPTs validated successfully")
    else:
        logger.warning(f"CPT validation issues: {validation['errors']}")
    
    if args.verbose:
        parameters.print_cpt_summary(model)
    
    logger.info(f"Parameter learning complete for {args.rank}")


def cmd_compare(args):
    logger.info("Comparing structures across ranks")
    
    # load all GES results
    graphs_by_rank = {}
    
    ranks_to_compare = args.ranks if args.ranks else ALL_RANKS
    
    for rank in ranks_to_compare:
        try:
            result = ges.load_ges_result(rank)
            graphs_by_rank[rank] = result['edges']
        except FileNotFoundError:
            logger.warning(f"GES result not found for {rank}. Skipping.")
            continue
    
    if len(graphs_by_rank) < 2:
        logger.error("Need at least 2 ranks to compare")
        return
    
    # compare
    comparison = compare.compare_edges(graphs_by_rank)
    
    # save comparison report
    output_file = REPORTS_DIR / "structure_comparison.md"
    compare.save_comparison_results(comparison, output_file)
    
    # create comparison table
    table = compare.create_edge_comparison_table(graphs_by_rank)
    table_file = REPORTS_DIR / "edge_comparison_table.csv"
    table.to_csv(table_file, index=False)
    logger.info(f"Saved edge comparison table to {table_file}")
    
    # create visualization
    if args.visualize:
        # get all variables
        all_vars = list(set(v for edges in graphs_by_rank.values() 
                           for v1, v2, _ in edges for v in [v1, v2]))
        
        viz_file = FIGURES_DIR / "rank_comparison.png"
        visualize.plot_rank_comparison(graphs_by_rank, all_vars, viz_file)
        
        # edge frequency plot
        freq_file = FIGURES_DIR / "edge_frequency.png"
        visualize.plot_edge_frequency(
            comparison['edge_frequency'],
            comparison['n_ranks'],
            freq_file
        )
    
    logger.info(f"Comparison complete. Report saved to {output_file}")


def cmd_query(args):
    logger.info(f"Running query for rank: {args.rank}")
    
    # load model
    try:
        model = parameters.load_bayesian_network(args.rank)
    except FileNotFoundError:
        logger.error(f"Bayesian network not found for {args.rank}. Run parameter learning first.")
        return
    
    if args.example:
        # run example queries
        from .config import EXAMPLE_QUERIES
        results = queries.run_example_queries(model, args.rank)
        
        # save results
        output_file = REPORTS_DIR / f"queries_{args.rank}.csv"
        results.to_csv(output_file, index=False)
        logger.info(f"Saved query results to {output_file}")
    
    elif args.evidence:
        # parse evidence
        evidence = {}
        for item in args.evidence.split(','):
            key, value = item.split('=')
            evidence[key.strip()] = value.strip()
        
        # run query
        prob = queries.p_win_given(evidence, model)
        
        print(f"\n{'='*60}")
        print(f"QUERY RESULT - {args.rank.upper()}")
        print(f"{'='*60}")
        print(f"Evidence: {evidence}")
        print(f"P(Win=1 | evidence) = {prob:.4f}")
        print(f"P(Win=0 | evidence) = {1-prob:.4f}")
        print(f"{'='*60}\n")
    
    else:
        logger.error("Specify either --example or --evidence")


def cmd_report(args):
    logger.info("Generating comprehensive report")
    
    from .config import EXAMPLE_QUERIES
    
    md = []
    md.append("# League of Legends Bayesian Network Analysis\n\n")
    md.append("## Structure Learning with GES\n\n")
    md.append("This report summarizes the Bayesian network structures learned from ")
    md.append("League of Legends match data across different rank tiers.\n\n")
    
    # for each rank
    for rank in ALL_RANKS:
        try:
            # load GES result
            ges_result = ges.load_ges_result(rank)
            
            md.append(f"### {rank}\n\n")
            md.append(f"**Number of edges:** {ges_result['n_edges']}\n\n")
            
            md.append("**Edges:**\n\n")
            for from_var, to_var, edge_type in sorted(ges_result['edges']):
                symbol = "→" if edge_type == "directed" else "—"
                md.append(f"- {from_var} {symbol} {to_var}\n")
            md.append("\n")
            
            # add figure
            fig_path = f"figures/cpdag_{rank}.png"
            md.append(f"![{rank} CPDAG]({fig_path})\n\n")
            
            # run queries if model exists
            try:
                model = parameters.load_bayesian_network(rank)
                
                md.append(f"#### Example Queries - {rank}\n\n")
                results = queries.query_multiple(EXAMPLE_QUERIES, model)
                md.append(results.to_markdown(index=False))
                md.append("\n\n")
            except FileNotFoundError:
                logger.warning(f"Model not found for {rank}. Skipping queries.")
        
        except FileNotFoundError:
            logger.warning(f"GES result not found for {rank}. Skipping.")
            continue
    
    # add comparison section
    md.append("## Structural Comparison\n\n")
    
    # read comparison file if it exists
    comparison_file = REPORTS_DIR / "structure_comparison.md"
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            md.append(f.read())
    else:
        md.append("*Run comparison analysis first (`python -m src.cli compare`)*\n\n")
    
    # save report
    output_file = REPORTS_DIR / "lol_ges_report.md"
    with open(output_file, 'w') as f:
        f.write("".join(md))
    
    logger.info(f"Report generated: {output_file}")


def cmd_full_pipeline(args):
    logger.info("Running full pipeline")
    
    ranks_to_process = args.ranks if args.ranks else ALL_RANKS
    
    for rank in ranks_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing rank: {rank}")
        logger.info(f"{'='*60}\n")
        
        # preprocessing
        logger.info("Step 1/4: Preprocessing")
        args.rank = rank
        args.sample_size = SAMPLE_SIZE
        cmd_preprocess(args)
        
        # structure learning
        logger.info("Step 2/4: Structure learning")
        args.no_constraints = False
        args.layout = "hierarchical"
        cmd_learn(args)
        
        # parameter learning
        logger.info("Step 3/4: Parameter learning")
        args.verbose = False
        cmd_params(args)
        
        # example queries
        logger.info("Step 4/4: Example queries")
        args.example = True
        args.evidence = None
        cmd_query(args)
    
    # compare
    logger.info("\nStep 5/5: Comparing structures")
    args.ranks = ranks_to_process
    args.visualize = True
    cmd_compare(args)
    
    # generate report
    logger.info("\nStep 6/5: Generating report")
    cmd_report(args)
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="League of Legends Bayesian Network Structure Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # preprocess command
    parser_preprocess = subparsers.add_parser('preprocess', help='Preprocess match data')
    parser_preprocess.add_argument(
        '--rank',
        type=str,
        default='all',
        help='Rank to preprocess (or "all")'
    )
    parser_preprocess.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for development (None = all data)'
    )
    
    # learn command
    parser_learn = subparsers.add_parser('learn', help='Learn graph structure with GES')
    parser_learn.add_argument('--rank', type=str, required=True, help='Rank to learn')
    parser_learn.add_argument('--no-constraints', action='store_true', help='Disable domain constraints')
    parser_learn.add_argument('--layout', type=str, default='hierarchical', 
                            choices=['hierarchical', 'spring'], help='Graph layout')
    
    # params command
    parser_params = subparsers.add_parser('params', help='Learn CPT parameters')
    parser_params.add_argument('--rank', type=str, required=True, help='Rank to learn')
    parser_params.add_argument('--verbose', action='store_true', help='Print CPT details')
    
    # compare command
    parser_compare = subparsers.add_parser('compare', help='Compare structures across ranks')
    parser_compare.add_argument('--ranks', nargs='+', help='Ranks to compare (default: all)')
    parser_compare.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    # query command
    parser_query = subparsers.add_parser('query', help='Run probabilistic queries')
    parser_query.add_argument('--rank', type=str, required=True, help='Rank to query')
    parser_query.add_argument('--example', action='store_true', help='Run example queries')
    parser_query.add_argument('--evidence', type=str, help='Evidence in format: Var1=val1,Var2=val2')
    
    # report command
    parser_report = subparsers.add_parser('report', help='Generate comprehensive report')
    
    # full pipeline command
    parser_full = subparsers.add_parser('full', help='Run full pipeline')
    parser_full.add_argument('--ranks', nargs='+', help='Ranks to process (default: all)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # dispatch to appropriate command
    try:
        if args.command == 'preprocess':
            cmd_preprocess(args)
        elif args.command == 'learn':
            cmd_learn(args)
        elif args.command == 'params':
            cmd_params(args)
        elif args.command == 'compare':
            cmd_compare(args)
        elif args.command == 'query':
            cmd_query(args)
        elif args.command == 'report':
            cmd_report(args)
        elif args.command == 'full':
            cmd_full_pipeline(args)
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()


