from pytorch_lightning.callbacks import Callback
import logging
import os
import datetime

logger = logging.getLogger(__name__)

class BenchmarkCallback(Callback):
    """PyTorch Lightning callback to evaluate benchmarks during training"""
    
    def __init__(self, benchmark_logger, eval_every_n_steps=5000, save_dir=None):
        """
        Initialize the benchmark callback
        
        Args:
            benchmark_logger: BenchmarkLogger instance
            eval_every_n_steps: How often to run benchmark evaluation
            save_dir: Directory to save benchmark results
        """
        super().__init__()
        self.benchmark_logger = benchmark_logger
        self.eval_every_n_steps = eval_every_n_steps
        self.save_dir = save_dir or './benchmark_results'
        self.last_eval_step = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize logging for this class
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add a file handler to capture benchmark logs
        log_file = os.path.join(self.save_dir, "benchmark_callback.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"BenchmarkCallback initialized with eval_every_n_steps={eval_every_n_steps}")
        self.logger.info(f"Benchmark results will be saved to {self.save_dir}")

    def on_train_start(self, trainer, pl_module):
        """Called when training begins"""
        if trainer.is_global_zero:
            self.logger.info("Training started")
            self.logger.info(f"Max steps: {trainer.max_steps}")
            
            # Verify benchmarks to run
            benchmarks = self.benchmark_logger.benchmarks_to_run
            self.logger.info(f"Benchmarks to run: {benchmarks}")
            
            # Verify save directory
            try:
                #test_file = os.path.join(self.save_dir, "test_file.txt")
                os.makedirs(self.save_dir, exist_ok=True)  # Ensure directory exists
                test_file = os.path.join(self.save_dir, f"test_file_rank{trainer.global_rank}.txt")
                with open(test_file, 'w') as f:
                    f.write("Test file to verify directory permissions\n")
                if os.path.exists(test_file):
                    self.logger.info(f"Successfully created test file in {self.save_dir}")
                    os.remove(test_file)
                else:
                    self.logger.error(f"Failed to verify test file creation in {self.save_dir}")
            except Exception as e:
                self.logger.error(f"Error verifying save directory: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Evaluate benchmarks periodically during training"""
        if trainer.is_global_zero:
            current_step = trainer.global_step
            
            #self.logger.info(f"Batch end at step {current_step}, last_eval_step: {self.last_eval_step}, threshold: {self.eval_every_n_steps}")
            
            # Check if it's time to evaluate based on the interval from config
            if (current_step - self.last_eval_step) >= self.eval_every_n_steps:
                self.logger.info(f"Starting benchmark evaluation at step {current_step}")
                
                # Save model's training state
                training = pl_module.training
                pl_module.eval()
                
                # Perform benchmark evaluation
                results = {}
                wandb_metrics = {}
                
                # Run eQTL benchmark
                if 'eqtl' in self.benchmark_logger.benchmarks_to_run:
                    try:
                        self.logger.info(f"Starting eQTL benchmark evaluation at step {current_step}")
                        
                        # Run the benchmark
                        eqtl_results = self.benchmark_logger.score_eqtl_benchmark()
                        self.logger.info(f"eQTL benchmark evaluation completed with results: {eqtl_results}")
                        
                        results.update(eqtl_results)
                        
                        # Extract key metrics for wandb
                        wandb_metrics.update({
                            'eqtl_nauprc': eqtl_results.get('eqtl_nauprc', 0.0),
                            'eqtl_cohens_d': eqtl_results.get('eqtl_cohens_d', 0.0),
                            'eqtl_nauprc_long': eqtl_results.get('eqtl_nauprc_long', 0.0),
                            'eqtl_cohens_d_long': eqtl_results.get('eqtl_cohens_d_long', 0.0)
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in eQTL benchmark evaluation: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                # Run rare variant benchmark
                if 'rare' in self.benchmark_logger.benchmarks_to_run:
                    try:
                        self.logger.info(f"Starting rare variant benchmark evaluation at step {current_step}")
                        rare_results = self.benchmark_logger.score_rare_variant_benchmark()
                        self.logger.info(f"Rare variant benchmark evaluation completed with results: {rare_results}")
                        
                        results.update(rare_results)
                        
                        # Extract key metrics for rare variants benchmark for wandb
                        for key, value in rare_results.items():
                            if key.startswith('rare_percentile_'):
                                wandb_metrics[key] = value
                        
                    except Exception as e:
                        self.logger.error(f"Error in rare variant benchmark evaluation: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                # Run TFBS benchmark
                if 'tfbs' in self.benchmark_logger.benchmarks_to_run:
                    try:
                        self.logger.info(f"Starting TFBS benchmark evaluation at step {current_step}")
                        tfbs_results = self.benchmark_logger.score_tfbs_benchmark()
                        self.logger.info(f"TFBS benchmark evaluation completed with results: {tfbs_results}")
                        
                        results.update(tfbs_results)
                        
                        # Extract key metrics for TFBS
                        wandb_metrics.update({
                            'tfbs_mean_delta_accuracy': tfbs_results.get('tfbs_mean_delta_accuracy', 0.0),
                            'tfbs_median_delta_accuracy': tfbs_results.get('tfbs_median_delta_accuracy', 0.0),
                            'tfbs_positive_fraction': tfbs_results.get('tfbs_positive_fraction', 0.0)
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in TFBS benchmark evaluation: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                # Save summary results
                if results:
                    try:
                        summary_path = os.path.join(self.save_dir, f"benchmark_summary_step_{current_step}.json")
                        self.logger.info(f"Saving benchmark summary to: {summary_path}")
                        
                        with open(summary_path, 'w') as f:
                            import json
                            json.dump(results, f, indent=2)
                        
                        # Verify file was created
                        if os.path.exists(summary_path):
                            self.logger.info(f"Successfully saved benchmark summary to {summary_path}")
                        else:
                            self.logger.error(f"Failed to save benchmark summary to {summary_path}")
                    except Exception as e:
                        self.logger.error(f"Error saving benchmark summary: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                # Log to wandb if metrics are available
                # Log to wandb if metrics are available
                if wandb_metrics:
                    try:
                        import wandb
                        if wandb.run is not None:
                            # Get the latest step from WandB
                            try:
                                current_wandb_step = wandb.run.step
                                # Use max to ensure we never go backward
                                log_step = max(current_step, current_wandb_step) 
                                self.logger.info(f"Logging metrics to wandb at step {log_step} (wandb step: {current_wandb_step}, trainer step: {current_step})")
                                wandb.log({f"benchmark/{k}": v for k, v in wandb_metrics.items()}, step=log_step)
                            except Exception as e:
                                self.logger.warning(f"Error synchronizing WandB step: {str(e)}")
                                # Fall back to just using run's current step (not trainer's)
                                wandb.log({f"benchmark/{k}": v for k, v in wandb_metrics.items()})
                        else:
                            # Try using trainer's logger
                            if hasattr(trainer, 'logger') and trainer.logger is not None:
                                self.logger.info(f"Logging key metrics via trainer.logger at step {current_step}")
                                for name, value in wandb_metrics.items():
                                    trainer.logger.log_metrics({f"benchmark/{name}": value}, step=current_step)
                    except Exception as e:
                        self.logger.warning(f"Could not log to wandb: {str(e)}")
                # line up here if 
                # Reset model's training state
                if training:
                    pl_module.train()
                
                # Update last evaluation step
                self.last_eval_step = current_step
                
                self.logger.info(f"Benchmark evaluation complete at step {current_step}")
            else:
                self.logger.debug(f"Skipping benchmark evaluation at step {current_step}")    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends"""
        if trainer.is_global_zero:  # Only run on main process
            self.logger.info("Training ended - doing final benchmark evaluation")
            
            try:
                # Create a plain text marker to indicate end of training
                import datetime
                end_marker_path = os.path.join(self.save_dir, "training_completed.txt")
                with open(end_marker_path, 'w') as f:
                    f.write(f"Training completed at step {trainer.global_step}\n")
                    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            except Exception as e:
                self.logger.error(f"Error creating training end marker: {str(e)}")
