"""
Distributed Training with Strategy Comparison

Improvements:
- CIFAR-10 dataset
- Comparison of multiple distribution strategies
- Performance metrics (training time, throughput)
- Model evaluation on test set
- Results saved to JSON
"""

import os
import json
import time
import logging
import tensorflow as tf

# Disable GPU for simulation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import dataset module
import cifar10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_strategy(strategy_name, strategy, epochs=5, steps_per_epoch=100):
    """
    Train model with given strategy and track performance
    
    Args:
        strategy_name: Name of the strategy
        strategy: TF distribution strategy object
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        
    Returns:
        dict: Results including time, accuracy, throughput
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with {strategy_name}")
    logger.info(f"{'='*60}")
    
    # Batch size configuration
    if strategy_name == "No Strategy":
        batch_size = 64
    else:
        per_worker_batch_size = 64
        num_replicas = strategy.num_replicas_in_sync
        batch_size = per_worker_batch_size * num_replicas
    
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Replicas: {strategy.num_replicas_in_sync if strategy else 1}")
    
    # Load dataset
    train_dataset, test_dataset = cifar10.cifar10_dataset(batch_size)
    
    # Build model within strategy scope
    if strategy:
        with strategy.scope():
            model = cifar10.build_and_compile_cnn_model()
    else:
        model = cifar10.build_and_compile_cnn_model()
    
    # Train and measure time
    logger.info("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    
    # Calculate throughput
    total_samples = steps_per_epoch * batch_size * epochs
    throughput = total_samples / training_time
    
    # Results
    results = {
        'strategy': strategy_name,
        'batch_size': batch_size,
        'num_replicas': strategy.num_replicas_in_sync if strategy else 1,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'training_time_seconds': round(training_time, 2),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'throughput_samples_per_sec': round(throughput, 2)
    }
    
    logger.info(f"\nResults:")
    logger.info(f"  Training time: {training_time:.2f}s")
    logger.info(f"  Test accuracy: {test_accuracy:.4f}")
    logger.info(f"  Throughput: {throughput:.2f} samples/sec")
    
    # Save model
    model_path = f'models/cifar10_{strategy_name.lower().replace(" ", "_")}.keras'
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    logger.info(f"  Model saved: {model_path}")
    
    return results


def compare_strategies():
    """
    Compare different distribution strategies
    """
    logger.info("="*60)
    logger.info("CIFAR-10 DISTRIBUTED TRAINING COMPARISON")
    logger.info("Author: Akhilesh Kasturi")
    logger.info("="*60)
    
    all_results = []
    
    # Strategy 1: No distribution (baseline)
    logger.info("\n[1/2] Baseline (No Distribution Strategy)")
    baseline_results = train_with_strategy(
        strategy_name="No Strategy",
        strategy=None,
        epochs=3,
        steps_per_epoch=100
    )
    all_results.append(baseline_results)
    
    # Strategy 2: MirroredStrategy (simulates multi-GPU on single machine)
    logger.info("\n[2/2] MirroredStrategy")
    mirrored_strategy = tf.distribute.MirroredStrategy()
    mirrored_results = train_with_strategy(
        strategy_name="MirroredStrategy",
        strategy=mirrored_strategy,
        epochs=3,
        steps_per_epoch=100
    )
    all_results.append(mirrored_results)
    
    # Save comparison results
    comparison = {
        'experiment': 'Distribution Strategy Comparison',
        'dataset': 'CIFAR-10',
        'model': 'CNN',
        'results': all_results,
        'winner': max(all_results, key=lambda x: x['test_accuracy'])['strategy']
    }
    
    with open('strategy_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*60)
    
    # Print comparison table
    logger.info("\nStrategy Comparison:")
    logger.info(f"{'Strategy':<20} {'Time(s)':<12} {'Test Acc':<12} {'Throughput':<15}")
    logger.info("-" * 60)
    for r in all_results:
        logger.info(
            f"{r['strategy']:<20} {r['training_time_seconds']:<12} "
            f"{r['test_accuracy']:<12.4f} {r['throughput_samples_per_sec']:<15.2f}"
        )
    
    logger.info(f"\nBest strategy: {comparison['winner']}")
    logger.info(f"Results saved to: strategy_comparison.json")


if __name__ == "__main__":
    # Disable TF warnings
    tf.get_logger().setLevel('ERROR')
    
    compare_strategies()