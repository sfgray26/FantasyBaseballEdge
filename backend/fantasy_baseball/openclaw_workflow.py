"""
OpenClaw Workflow Manager

Manages daily operations for OpenClaw:
1. Morning briefing generation and email delivery
2. Daily token usage tracking and reporting
3. Circuit breaker monitoring
4. Performance metrics collection
5. Weekly summary generation

This integrates with the coordinator to provide:
- Automated daily emails (when configured)
- Health monitoring
- Cost tracking and alerting
- Performance optimization recommendations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from backend.fantasy_baseball.openclaw_email_notifier import (
    send_daily_briefing,
    send_budget_warning,
    send_circuit_breaker_alert,
)

logger = logging.getLogger("openclaw.workflow")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_LOG = Path(".openclaw/token-usage.jsonl")


class OpenClawWorkflowManager:
    """Manages daily OpenClaw workflows and reporting."""
    
    def __init__(self):
        self.stats_cache: Dict = {}
        self.last_stats_reset = datetime.now().date()
    
    def _read_token_usage(self, days: int = 1) -> List[Dict]:
        """Read token usage from log file."""
        entries = []
        if not TOKEN_LOG.exists():
            return entries
        
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            with open(TOKEN_LOG, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(entry.get('timestamp', '2000-01-01'))
                        if entry_time >= cutoff:
                            entries.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Error reading token log: {e}")
        
        return entries
    
    def collect_stats(self) -> Dict:
        """Collect comprehensive stats for reporting."""
        # Read last 24 hours of usage
        entries = self._read_token_usage(days=1)
        
        if not entries:
            return {
                'circuit_breaker': 'UNKNOWN',
                'daily_cost_usd': 0.0,
                'budget_remaining_pct': 100.0,
                'tasks_24h': 0,
                'tasks_local': 0,
                'tasks_kimi': 0,
                'latency_local_ms': 0,
                'latency_kimi_ms': 0,
                'events': []
            }
        
        # Calculate stats
        local_entries = [e for e in entries if e.get('engine') == 'local']
        kimi_entries = [e for e in entries if e.get('engine') == 'kimi']
        
        total_cost = sum(e.get('estimated_cost_usd', 0) for e in entries)
        
        avg_latency_local = (
            sum(e.get('latency_ms', 0) for e in local_entries) / len(local_entries)
            if local_entries else 0
        )
        
        avg_latency_kimi = (
            sum(e.get('latency_ms', 0) for e in kimi_entries) / len(kimi_entries)
            if kimi_entries else 0
        )
        
        # Budget calculation (from config)
        daily_budget = 5.00  # Default from config.yaml
        budget_remaining = max(0, 100 - (total_cost / daily_budget * 100))
        
        # Collect events
        events = []
        
        # Check for high-stakes escalations
        high_stakes = [e for e in entries if e.get('task_type') == 'high_stakes_integrity']
        if high_stakes:
            events.append(f"{len(high_stakes)} high-stakes escalation(s)")
        
        # Check for VOLATILE verdicts
        volatile = [e for e in entries if 'VOLATILE' in str(e.get('output', ''))]
        if volatile:
            events.append(f"{len(volatile)} VOLATILE verdict(s)")
        
        # Check for circuit breaker events (would need to track these separately)
        
        stats = {
            'circuit_breaker': 'CLOSED',  # Would need to query actual state
            'daily_cost_usd': total_cost,
            'budget_remaining_pct': budget_remaining,
            'tasks_24h': len(entries),
            'tasks_local': len(local_entries),
            'tasks_kimi': len(kimi_entries),
            'latency_local_ms': avg_latency_local,
            'latency_kimi_ms': avg_latency_kimi,
            'events': events if events else ['Normal operations']
        }
        
        self.stats_cache = stats
        return stats
    
    async def run_daily_briefing(self) -> bool:
        """Generate and send daily briefing."""
        logger.info("Generating daily briefing...")
        
        stats = self.collect_stats()
        
        # Send email
        result = send_daily_briefing(stats)
        
        # Check if budget warning needed
        if stats['budget_remaining_pct'] < 20:
            send_budget_warning(100 - stats['budget_remaining_pct'])
        
        logger.info(f"Daily briefing sent: {result}")
        return result
    
    def generate_weekly_summary(self) -> str:
        """Generate weekly performance summary."""
        entries = self._read_token_usage(days=7)
        
        if not entries:
            return "No data available for weekly summary."
        
        # Calculate weekly stats
        local_entries = [e for e in entries if e.get('engine') == 'local']
        kimi_entries = [e for e in entries if e.get('engine') == 'kimi']
        
        total_cost = sum(e.get('estimated_cost_usd', 0) for e in entries)
        total_tasks = len(entries)
        
        # Success rate
        successful = sum(1 for e in entries if e.get('success', False))
        success_rate = (successful / total_tasks * 100) if total_tasks else 0
        
        summary = f"""
WEEKLY OPENCLAW SUMMARY
{'='*50}
Week of: {(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}

TASKS PROCESSED: {total_tasks}
- Local (qwen2.5:3b): {len(local_entries)} ({len(local_entries)/total_tasks*100:.1f}%)
- Kimi Escalations: {len(kimi_entries)} ({len(kimi_entries)/total_tasks*100:.1f}%)

SUCCESS RATE: {success_rate:.1f}%

COST ANALYSIS:
- Total Cost: ${total_cost:.2f}
- Avg Cost/Task: ${total_cost/total_tasks:.3f}
- Local Savings: ~${len(local_entries) * 0.05:.2f} (vs all Kimi)

PERFORMANCE:
- Avg Local Latency: {sum(e.get('latency_ms', 0) for e in local_entries) / len(local_entries):.0f}ms
- Avg Kimi Latency: {sum(e.get('latency_ms', 0) for e in kimi_entries) / len(kimi_entries) if kimi_entries else 0:.0f}ms

ROUTING EFFICIENCY:
{len(local_entries)/total_tasks*100:.1f}% of tasks handled locally (cost-efficient)
"""
        
        return summary
    
    async def run_health_check(self) -> Dict:
        """Run health check on OpenClaw components."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check token log accessibility
        results['checks']['token_log'] = {
            'status': 'OK' if TOKEN_LOG.exists() else 'MISSING',
            'path': str(TOKEN_LOG)
        }
        
        # Check cache directory
        results['checks']['cache_dir'] = {
            'status': 'OK' if CACHE_DIR.exists() else 'ERROR',
            'path': str(CACHE_DIR)
        }
        
        # Check email configuration
        from backend.fantasy_baseball.openclaw_email_notifier import get_notifier
        notifier = get_notifier()
        results['checks']['email_config'] = {
            'status': 'CONFIGURED' if notifier.enabled else 'DISABLED',
            'enabled': notifier.enabled
        }
        
        # Overall status
        all_ok = all(c['status'] in ('OK', 'CONFIGURED', 'DISABLED') for c in results['checks'].values())
        results['overall'] = 'HEALTHY' if all_ok else 'DEGRADED'
        
        return results


# Additional OpenClaw capabilities
class OpenClawEnhancements:
    """Additional capabilities for OpenClaw."""
    
    @staticmethod
    def export_daily_report(output_path: Optional[Path] = None) -> Path:
        """Export detailed daily report to file."""
        if output_path is None:
            output_path = CACHE_DIR / f"openclaw_report_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        manager = OpenClawWorkflowManager()
        stats = manager.collect_stats()
        
        report = {
            'date': datetime.now().isoformat(),
            'stats': stats,
            'config': {
                'daily_budget_usd': 5.00,
                'circuit_breaker_threshold': 5,
                'circuit_breaker_timeout': 60,
                'max_concurrent': 8
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Daily report exported: {output_path}")
        return output_path
    
    @staticmethod
    def generate_optimization_recommendations() -> List[str]:
        """Generate recommendations for optimizing OpenClaw performance."""
        manager = OpenClawWorkflowManager()
        stats = manager.collect_stats()
        recommendations = []
        
        # Check routing efficiency
        if stats['tasks_24h'] > 0:
            local_pct = stats['tasks_local'] / stats['tasks_24h'] * 100
            if local_pct < 70:
                recommendations.append(
                    f"Routing efficiency low ({local_pct:.1f}% local). "
                    "Consider adjusting escalation thresholds to reduce costs."
                )
            elif local_pct > 95 and stats['tasks_24h'] > 50:
                recommendations.append(
                    f"High local routing ({local_pct:.1f}%). "
                    "Verify quality gates aren't too permissive."
                )
        
        # Check latency
        if stats['latency_local_ms'] > 1000:
            recommendations.append(
                f"Local LLM latency high ({stats['latency_local_ms']:.0f}ms). "
                "Check Ollama service resources."
            )
        
        # Check budget
        if stats['budget_remaining_pct'] < 30:
            recommendations.append(
                f"Budget running low ({stats['budget_remaining_pct']:.0f}% remaining). "
                "Consider increasing daily limit or reducing high-stakes routing."
            )
        
        # Check for repeated patterns
        if not recommendations:
            recommendations.append("System operating within normal parameters.")
        
        return recommendations


# Convenience functions
async def run_morning_briefing():
    """Run the morning briefing workflow."""
    manager = OpenClawWorkflowManager()
    return await manager.run_daily_briefing()


def get_optimization_recommendations() -> List[str]:
    """Get optimization recommendations."""
    return OpenClawEnhancements.generate_optimization_recommendations()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("OpenClaw Workflow Manager Test")
    print("=" * 50)
    
    # Test stats collection
    manager = OpenClawWorkflowManager()
    stats = manager.collect_stats()
    print("\nCurrent Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test recommendations
    print("\nOptimization Recommendations:")
    for rec in get_optimization_recommendations():
        print(f"  - {rec}")
    
    # Test weekly summary
    print("\nWeekly Summary:")
    print(manager.generate_weekly_summary())
    
    # Export report
    report_path = OpenClawEnhancements.export_daily_report()
    print(f"\nReport exported: {report_path}")
