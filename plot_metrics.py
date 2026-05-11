import json
import matplotlib
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List
from bat.logging import create_logger

logger = create_logger(__name__, level="info")
# Base output directory (resolved from this file location)
OUTPUT_DIR = (Path(__file__).resolve().parent / "../../output").resolve()


def extract_base_task_id(task_id: str) -> str:
    """Remove __tryN suffix from task_id"""
    return re.sub(r'__try\d+$', '', task_id)


def group_episodes_by_task(per_episode: List[dict]) -> Dict[str, List[dict]]:
    """Group episodes by base task_id (without __tryN suffix)"""
    grouped: Dict[str, List[dict]] = {}
    for episode in per_episode:
        base_id = extract_base_task_id(episode['task_id'])
        grouped.setdefault(base_id, []).append(episode)
    return grouped


def average_episodes(episodes: List[dict]) -> dict:
    """Average metrics across multiple attempts of the same task"""
    if len(episodes) == 1:
        return episodes[0]
    
    n = len(episodes)
    
    # Average time
    avg_time = sum(ep['time']['wall_ms'] for ep in episodes) / n
    
    # Average tokens
    avg_prompt = sum(ep['tokens']['prompt_tokens'] for ep in episodes) / n
    avg_completion = sum(ep['tokens']['completion_tokens'] for ep in episodes) / n
    avg_total = sum(ep['tokens']['total_tokens'] for ep in episodes) / n
    
    # Success rate (fraction of successful attempts)
    success_rate = sum(ep['success'] for ep in episodes) / n
    
    result = {
        'task_id': extract_base_task_id(episodes[0]['task_id']),
        'status': episodes[0]['status'],
        'success': success_rate >= 0.5,  # Majority vote
        'success_rate': success_rate,
        'attempts': n,
        'time': {'wall_ms': avg_time},
        'tokens': {
            'prompt_tokens': avg_prompt,
            'completion_tokens': avg_completion,
            'total_tokens': avg_total,
        }
    }
    
    # Average qualitative if available
    qual_episodes = [ep for ep in episodes if 'qualitative' in ep]
    if qual_episodes:
        # Filter out None values when averaging
        relevance_vals = [ep['qualitative']['response_relevance'] for ep in qual_episodes if ep['qualitative'].get('response_relevance') is not None]
        completion_vals = [ep['qualitative']['task_completion_quality'] for ep in qual_episodes if ep['qualitative'].get('task_completion_quality') is not None]
        hallucination_vals = [ep['qualitative']['hallucination_score'] for ep in qual_episodes if ep['qualitative'].get('hallucination_score') is not None]
        
        result['qualitative'] = {
            'response_relevance': sum(relevance_vals) / len(relevance_vals) if relevance_vals else 0,
            'task_completion_quality': sum(completion_vals) / len(completion_vals) if completion_vals else 0,
            'hallucination_score': sum(hallucination_vals) / len(hallucination_vals) if hallucination_vals else 0,
        }
    
    return result


def get_per_episode_averages(metrics_data: dict) -> List[dict]:
    """Get per-episode metrics, averaging across attempts when k>1"""
    per_episode = metrics_data.get('per_episode', [])
    if not per_episode:
        return []
    
    grouped = group_episodes_by_task(per_episode)
    return [average_episodes(episodes) for episodes in grouped.values()]


def load_metrics(task_id: str = None) -> Dict[str, dict]:
    """Load metrics.json files from OUTPUT_DIR.

    If task_id is provided, only load files under OUTPUT_DIR/task_id/*/metrics.json.
    Otherwise, load files under OUTPUT_DIR/*/*/metrics.json.
    """
    metrics = {}

    pattern = f"{task_id}/*/metrics.json" if task_id else "*/*/metrics.json"
    for metrics_file in OUTPUT_DIR.glob(pattern):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics[metrics_file.parent.name] = json.load(f)

    return metrics


def plot_per_episode_comparison(metrics: Dict[str, dict]):
    """Create per-task comparison charts showing all models for each task"""
    
    if not metrics:
        return []
    
    # First, collect all tasks and their metrics per model
    tasks_by_model = {}
    all_task_ids = set()
    
    for run_name, data in metrics.items():
        episodes = get_per_episode_averages(data)
        tasks_by_model[run_name] = {ep['task_id']: ep for ep in episodes}
        all_task_ids.update(ep['task_id'] for ep in episodes)
    
    if not all_task_ids:
        return []
    
    # Sort task IDs for consistent ordering
    sorted_tasks = sorted(all_task_ids)
    model_names = list(metrics.keys())
    
    # Check if any model has qualitative metrics
    has_qualitative = any(
        any('qualitative' in ep for ep in get_per_episode_averages(data))
        for data in metrics.values()
    )
    
    figures = []
    
    # Create a figure for each task
    for task_id in sorted_tasks:
        # Collect data from all models for this task
        task_data = []
        available_models = []
        
        for run_name in model_names:
            if task_id in tasks_by_model[run_name]:
                task_data.append(tasks_by_model[run_name][task_id])
                available_models.append(run_name)
        
        if not task_data:
            continue
        
        display_names = [name.replace("hermes_benchmark_", "").replace("ollama-", "") for name in available_models]
        
        # Create figure with subplots (removed success rate chart)
        n_plots = 3 if has_qualitative else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(max(10, len(task_data)*0.8), 4*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle(f'Task: {task_id}', fontsize=14, fontweight='bold')
        
        # 1. Time per model
        times = [ep['time']['wall_ms'] / 1000 for ep in task_data]
        axes[0].bar(range(len(display_names)), times, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Time (s)', fontsize=11)
        axes[0].set_title('Execution Time by Model', fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(len(display_names)))
        axes[0].set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(axes[0].patches, times):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 2. Tokens per model
        prompt_tokens = [ep['tokens']['prompt_tokens'] for ep in task_data]
        completion_tokens = [ep['tokens']['completion_tokens'] for ep in task_data]
        x = range(len(display_names))
        axes[1].bar(x, prompt_tokens, label='Prompt', color='cornflowerblue', alpha=0.8)
        axes[1].bar(x, completion_tokens, bottom=prompt_tokens, 
                   label='Completion', color='lightcoral', alpha=0.8)
        axes[1].set_ylabel('Tokens', fontsize=11)
        axes[1].set_title('Token Usage by Model', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        axes[1].legend(fontsize=9)
        axes[1].grid(axis='y', alpha=0.3)
        
        # 3. Qualitative metrics (if available)
        if has_qualitative:
            x_pos = range(len(display_names))
            width = 0.25
            
            relevance = [ep.get('qualitative', {}).get('response_relevance', 0) for ep in task_data]
            completion_q = [ep.get('qualitative', {}).get('task_completion_quality', 0) for ep in task_data]
            hallucination = [ep.get('qualitative', {}).get('hallucination_score', 0) for ep in task_data]
            
            x_rel = [i - width for i in x_pos]
            x_comp = x_pos
            x_hall = [i + width for i in x_pos]
            
            axes[2].bar(x_rel, relevance, width, label='Relevance', color='lightblue', alpha=0.8)
            axes[2].bar(x_comp, completion_q, width, label='Completion', color='lightgreen', alpha=0.8)
            axes[2].bar(x_hall, hallucination, width, label='Groundedness', color='khaki', alpha=0.8)
            
            axes[2].set_ylabel('Score', fontsize=11)
            axes[2].set_title('Qualitative Metrics by Model', fontsize=12, fontweight='bold')
            axes[2].set_xticks(x_pos)
            axes[2].set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
            axes[2].set_ylim(0, 1.1)
            axes[2].legend(fontsize=9)
            axes[2].grid(axis='y', alpha=0.3)
            axes[2].axhline(y=0.7, color='orange', linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        
        # Use sanitized task name for filename
        safe_task = task_id.replace(':', '-').replace('/', '-').replace(' ', '_')
        figures.append((f'per_task_{safe_task}', fig))
    
    return figures


def plot_comparison(metrics: Dict[str, dict], task_id: str = ""):
    """Create comparison charts between different runs
    
    Args:
        metrics: Dictionary of run metrics
        task_id: Optional task_id to include in filename
    """
    
    if not metrics:
        logger.error("No metrics.json file found!")
        return
    
    # Extract run names and data
    run_names = list(metrics.keys())
    display_names = list(map(lambda x : x.replace("hermes_get_delete_", ""), run_names))  # Shorten names for display if needed
    
    # Check if any run has qualitative metrics
    has_qualitative = any("qualitative" in m.get("summary", {}) for m in metrics.values())
    
    # Prepare data
    times = []
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    
    # Qualitative metrics
    relevance_scores = []
    completion_quality_scores = []
    hallucination_scores = []
    
    for name in run_names:
        summary = metrics[name].get("summary", {})
        
        # Time (convert ms to seconds)
        times.append(summary.get("time", {}).get("total_wall_ms", 0) / 1000)
                
        # Tokens
        tokens = summary.get("tokens", {})
        prompt_tokens.append(tokens.get("prompt_tokens_total", 0))
        completion_tokens.append(tokens.get("completion_tokens_total", 0))
        total_tokens.append(tokens.get("total_tokens_total", 0))
        
        # Qualitative
        qual = summary.get("qualitative", {})
        relevance_scores.append(qual.get("response_relevance", {}).get("avg", None))
        completion_quality_scores.append(qual.get("task_completion_quality", {}).get("avg", None))
        hallucination_scores.append(qual.get("hallucination_score", {}).get("avg", None))
    
    # Create separate figures for each metric
    figures = []
    
    # 1. Total time
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    bars1 = ax1.bar(range(len(run_names)), times, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Total Execution Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(run_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    fig1.tight_layout()
    figures.append(('execution_time', fig1))

    # 2. Time (up) vs Total Tokens (down) — mirror bar chart
    fig2b, ax2b = plt.subplots(figsize=(max(10, len(run_names) * 1.2), 7))
    x2b = range(len(run_names))

    # Normalize both series to the same scale so bars are comparable
    max_time = max(times) if max(times) > 0 else 1
    max_tok = max(total_tokens) if max(total_tokens) > 0 else 1
    times_norm = [t / max_time for t in times]
    tokens_norm = [-t / max_tok for t in total_tokens]  # negative → downward

    ax2b.bar(x2b, times_norm, color='steelblue', alpha=0.75, label='Execution Time')
    ax2b.bar(x2b, tokens_norm, color='darkorange', alpha=0.75, label='Total Tokens')
    ax2b.axhline(0, color='black', linewidth=0.8)

    # Annotate actual values above/below bars
    for xi, t_n, t_val, tok_n, tok_val in zip(x2b, times_norm, times, tokens_norm, total_tokens):
        ax2b.text(xi, t_n + 0.02, f'{t_val:.1f}s', ha='center', va='bottom',
                  fontsize=8, color='steelblue')
        ax2b.text(xi, tok_n - 0.02, f'{tok_val:,}', ha='center', va='top',
                  fontsize=8, color='darkorange')

    ax2b.set_xticks(x2b)
    ax2b.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax2b.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2b.set_yticklabels([f'max\n({max_tok:,} tok)', '50%', '0',
                           '50%', f'max\n({max_time:.1f}s)'], fontsize=8)
    ax2b.set_title('Execution Time ↑  vs  Total Tokens ↓', fontsize=14, fontweight='bold')
    ax2b.legend(fontsize=9)
    ax2b.grid(axis='y', alpha=0.2)

    fig2b.tight_layout()
    figures.append(('time_vs_total_tokens', fig2b))

    
    # 3. Token Usage (stacked bar)
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    x = range(len(run_names))
    ax3.bar(x, prompt_tokens, label='Prompt Tokens', color='cornflowerblue', alpha=0.8)
    ax3.bar(x, completion_tokens, bottom=prompt_tokens, 
            label='Completion Tokens', color='lightcoral', alpha=0.8)
    ax3.set_xlabel('Run')
    ax3.set_ylabel('Token Count')
    ax3.set_title('Token Usage (Prompt vs Completion)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    fig3.tight_layout()
    figures.append(('token_usage', fig3))
    
    # 4. Total Tokens
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    bars4 = ax4.bar(range(len(run_names)), total_tokens, color='mediumpurple', alpha=0.7)
    ax4.set_xlabel('Run')
    ax4.set_ylabel('Total Tokens')
    ax4.set_title('Total Tokens', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(run_names)))
    ax4.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars4, total_tokens):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    
    fig4.tight_layout()
    figures.append(('total_tokens', fig4))
    
    # 5-6. Qualitative metrics (if available)
    if has_qualitative:
        # Plot qualitative scores
        fig5 = plt.figure(figsize=(10, 6))
        ax5 = fig5.add_subplot(111)
        x_pos = range(len(run_names))
        width = 0.25
        
        # Filter out None values for plotting
        rel_vals = [v if v is not None else 0 for v in relevance_scores]
        comp_vals = [v if v is not None else 0 for v in completion_quality_scores]
        hall_vals = [v if v is not None else 0 for v in hallucination_scores]
        
        x_rel = [i - width for i in x_pos]
        x_comp = x_pos
        x_hall = [i + width for i in x_pos]
        
        ax5.bar(x_rel, rel_vals, width, label='Response Relevance', color='lightblue', alpha=0.8)
        ax5.bar(x_comp, comp_vals, width, label='Task Completion', color='lightgreen', alpha=0.8)
        ax5.bar(x_hall, hall_vals, width, label='Groundedness (1-halluc)', color='khaki', alpha=0.8)
        
        ax5.set_xlabel('Run')
        ax5.set_ylabel('Score (0-1)')
        ax5.set_title('Qualitative Metrics (LLM Judge)', fontsize=14, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
        ax5.set_ylim(0, 1.1)
        ax5.legend(fontsize=8)
        ax5.grid(axis='y', alpha=0.3)
        ax5.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='Good threshold')
        
        fig5.tight_layout()
        figures.append(('qualitative_metrics', fig5))
        
        # Separate charts for each qualitative metric
        # 6. Response Relevance
        fig6 = plt.figure(figsize=(10, 6))
        ax6 = fig6.add_subplot(111)
        rel_vals = [v if v is not None else 0 for v in relevance_scores]
        colors_rel = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in rel_vals]
        bars6 = ax6.bar(range(len(run_names)), rel_vals, color=colors_rel, alpha=0.7)
        ax6.set_xlabel('Run')
        ax6.set_ylabel('Score (0-1)')
        ax6.set_title('Response Relevance', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(run_names)))
        ax6.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
        ax6.set_ylim(0, 1.1)
        ax6.grid(axis='y', alpha=0.3)
        ax6.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax6.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        
        for bar, val in zip(bars6, rel_vals):
            if val > 0:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        fig6.tight_layout()
        figures.append(('response_relevance', fig6))
        
        # 7. Task Completion Quality
        fig7 = plt.figure(figsize=(10, 6))
        ax7 = fig7.add_subplot(111)
        comp_vals = [v if v is not None else 0 for v in completion_quality_scores]
        colors_comp = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in comp_vals]
        bars7 = ax7.bar(range(len(run_names)), comp_vals, color=colors_comp, alpha=0.7)
        ax7.set_xlabel('Run')
        ax7.set_ylabel('Score (0-1)')
        ax7.set_title('Task Completion Quality', fontsize=14, fontweight='bold')
        ax7.set_xticks(range(len(run_names)))
        ax7.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
        ax7.set_ylim(0, 1.1)
        ax7.grid(axis='y', alpha=0.3)
        ax7.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax7.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        
        for bar, val in zip(bars7, comp_vals):
            if val > 0:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        fig7.tight_layout()
        figures.append(('task_completion_quality', fig7))
        
        # 8. Hallucination Score (Groundedness)
        fig8 = plt.figure(figsize=(10, 6))
        ax8 = fig8.add_subplot(111)
        hall_vals = [v if v is not None else 0 for v in hallucination_scores]
        colors_hall = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in hall_vals]
        bars8 = ax8.bar(range(len(run_names)), hall_vals, color=colors_hall, alpha=0.7)
        ax8.set_xlabel('Run')
        ax8.set_ylabel('Score (0-1)')
        ax8.set_title('Groundedness (Hallucination Score)', fontsize=14, fontweight='bold')
        ax8.set_xticks(range(len(run_names)))
        ax8.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
        ax8.set_ylim(0, 1.1)
        ax8.grid(axis='y', alpha=0.3)
        ax8.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax8.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        
        for bar, val in zip(bars8, hall_vals):
            if val > 0:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        fig8.tight_layout()
        figures.append(('hallucination_score', fig8))
    
    # Save all figures
    logger.debug("\nSaved charts:")
    for name, fig in figures:
        if task_id:
            chart_dir = OUTPUT_DIR / task_id
            chart_dir.mkdir(parents=True, exist_ok=True)
        else:
            chart_dir = OUTPUT_DIR
            chart_dir.mkdir(parents=True, exist_ok=True)
        output_path = chart_dir / f"metrics_{name}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.debug(f"  - {output_path}")
    
    # Show all figures only if not using Agg backend
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        # Close figures to free memory when not showing
        for _, fig in figures:
            plt.close(fig)


def print_summary_table(metrics: Dict[str, dict]):
    """Print a textual summary table"""
    # Check if any run has qualitative metrics
    has_qualitative = any("qualitative" in m.get("summary", {}) for m in metrics.values())
    
    logger.info("\n" + "="*150)
    logger.info("SUMMARY TABLE")
    logger.info("="*150)
    
    if has_qualitative:
        logger.info(f"{'Run Name':<40} {'k':<4} {'Time(s)':<10} {'Pass%':<8} {'Tokens':<12} {'Relevance':<10} {'Completion':<11} {'Groundedness':<12}")
        logger.info("-"*150)
    else:
        logger.info(f"{'Run Name':<40} {'k':<4} {'Time(s)':<12} {'Pass%':<8} {'Prompt':<10} {'Compl':<10} {'Total':<10}")
        logger.info("-"*110)
    
    for name, data in metrics.items():
        summary = data.get("summary", {})
        k = summary.get("k_attempts", 1)
        time_s = summary.get("time", {}).get("total_wall_ms", 0) / 1000
        pass_rate = summary.get("pass_rate", 0) * 100
        tokens = summary.get("tokens", {})
        total = tokens.get("total_tokens_total", 0)
        
        if has_qualitative:
            qual = summary.get("qualitative", {})
            relevance = qual.get("response_relevance", {}).get("avg", None)
            completion = qual.get("task_completion_quality", {}).get("avg", None)
            groundedness = qual.get("hallucination_score", {}).get("avg", None)
            
            rel_str = f"{relevance:.3f}" if relevance is not None else "N/A"
            comp_str = f"{completion:.3f}" if completion is not None else "N/A"
            ground_str = f"{groundedness:.3f}" if groundedness is not None else "N/A"
            
            logger.info(f"{name:<40} {k:<4} {time_s:<10.2f} {pass_rate:<8.1f} {total:<12,} {rel_str:<10} {comp_str:<11} {ground_str:<12}")
        else:
            prompt = tokens.get("prompt_tokens_total", 0)
            completion_tok = tokens.get("completion_tokens_total", 0)
            logger.info(f"{name:<40} {k:<4} {time_s:<12.2f} {pass_rate:<8.1f} {prompt:<10,} {completion_tok:<10,} {total:<10,}")
    
    if has_qualitative:
        logger.info("="*150)
    else:
        logger.info("="*110)
    
    logger.info("\nNote: k = number of attempts per task. If k>1, metrics show ALL attempts.")
    logger.info()


if __name__ == "__main__":
    import sys
    
    # Check for task_id argument
    task_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if task_id:
        logger.info(f"Loading metrics for task_id: {task_id}...")
    else:
        logger.info("Loading all metrics...")
    
    metrics = load_metrics(task_id=task_id)
    
    if metrics:
        logger.info(f"Found {len(metrics)} runs")
        print_summary_table(metrics)
        
        logger.info("\nGenerating charts...")
        
        # 1. Overall comparison charts (across models)
        logger.info("  - Overall comparison charts...")
        plot_comparison(metrics, task_id=task_id or "")
        
        # 2. Per-task charts (comparing all models for each task)
        logger.info("  - Per-task comparison charts...")
        per_task_figs = plot_per_episode_comparison(metrics)
        
        # Save per-task figures
        if per_task_figs:
            logger.info(f"\nSaved {len(per_task_figs)} per-task charts:")
            for name, fig in per_task_figs:
                if task_id:
                    chart_dir = OUTPUT_DIR / task_id
                    chart_dir.mkdir(parents=True, exist_ok=True)
                else:
                    chart_dir = OUTPUT_DIR
                    chart_dir.mkdir(parents=True, exist_ok=True)
                output_path = chart_dir / f"metrics_{name}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"  - {output_path}")
        
        logger.info("\nAll charts generated!")
        
        # Show charts only if not using Agg backend
        if matplotlib.get_backend() != 'Agg':
            plt.show()
        else:
            # Close all figures to free memory
            plt.close('all')
    else:
        if task_id:
            logger.error(f"No metrics found for task_id: {task_id}")
        else:
            logger.error("No metrics found in OUTPUT_DIR")
