# src/main.py

#!/usr/bin/env python3
"""
Cup Tracker Pro - Professional Self-Learning Cup Tracking System
Main application entry point with advanced features for Arch Linux
"""

import asyncio
import signal
import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import json
import argparse
from typing import Optional
import torch
import numpy as np

from core.neural_tracker import NeuralCupTracker
from core.adaptive_learner import AdaptiveLearner
from core.motion_predictor import MotionPredictor
from vision.enhanced_detector import EnhancedDetector
from ui.wayland_overlay import WaylandOverlay
from ui.gtk_control import GTKControlPanel
from utils.arch_integration import ArchSystemIntegration
from utils.performance_monitor import PerformanceMonitor
from utils.telemetry import TelemetryCollector

# Configure logging
log_file = Path.home() / '.local' / 'share' / 'cup-tracker' / 'cup-tracker.log'
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CupTrackerPro:
    """Main application class"""
    
    def __init__(self, config_path: Optional[Path] = None):
        logger.info("Initializing Cup Tracker Pro")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.system = ArchSystemIntegration()
        self.performance = PerformanceMonitor()
        self.telemetry = TelemetryCollector()
        
        # Core vision components
        self.detector = EnhancedDetector(self.config)
        self.tracker = NeuralCupTracker(
            model_path=Path(self.config['model_path']),
            device=self.config.get('device', 'cuda')
        )
        
        # Learning system
        self.learner = AdaptiveLearner(Path(self.config['model_dir']))
        self.predictor = MotionPredictor()
        
        # UI components
        self.overlay = None
        self.control_panel = None
        
        # State
        self.running = False
        self.tracking_active = False
        self.current_mode = 'idle'
        
        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.last_frame_time = 0
        
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from file"""
        default_config = {
            'model_path': '/usr/share/cup-tracker/models/siamese.pth',
            'model_dir': str(Path.home() / '.local' / 'share' / 'cup-tracker' / 'models'),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'capture_fps': 60,
            'detection_interval': 5,
            'learning_rate': 0.001,
            'telemetry_enabled': True,
            'performance_logging': True,
            'wayland_overlay': True,
            'adaptive_tracking': True
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            # Save default config
            config_path = config_path or Path('/etc/cup-tracker/config.json')
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default configuration at {config_path}")
        
        return default_config
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing components...")
        
        # Check system requirements
        self.system.check_requirements()
        
        # Get display information
        display_info = self.system.get_display_info()
        
        # Initialize overlay
        if self.config['wayland_overlay']:
            self.overlay = WaylandOverlay(
                display_info['width'],
                display_info['height']
            )
            await self.overlay.initialize()
        
        # Initialize control panel
        self.control_panel = GTKControlPanel(self)
        await self.control_panel.initialize()
        
        # Start performance monitoring
        if self.config['performance_logging']:
            await self.performance.start()
        
        # Start telemetry
        if self.config['telemetry_enabled']:
            await self.telemetry.start()
        
        logger.info("All components initialized successfully")
        
    async def run(self):
        """Main application loop"""
        self.running = True
        
        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        logger.info("Cup Tracker Pro started")
        
        # Main processing loop
        while self.running:
            try:
                frame_start = asyncio.get_event_loop().time()
                
                # Process based on mode
                if self.current_mode == 'tracking':
                    await self._process_tracking()
                elif self.current_mode == 'learning':
                    await self._process_learning()
                
                # Update FPS counter
                self.frame_count += 1
                elapsed = asyncio.get_event_loop().time() - frame_start
                self.fps = 1.0 / elapsed if elapsed > 0 else 0
                
                # Adaptive delay based on performance
                target_frame_time = 1.0 / self.config['capture_fps']
                if elapsed < target_frame_time:
                    await asyncio.sleep(target_frame_time - elapsed)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def _process_tracking(self):
        """Process tracking mode"""
        # Capture frame
        frame = await self.system.capture_frame()
        if frame is None:
            return
        
        # Detect cups
        cups = await self.detector.detect(frame)
        
        # Track target
        if self.tracking_active:
            state = await self.tracker.track(frame)
            
            if state:
                # Update overlay
                if self.overlay:
                    await self.overlay.update_overlay({
                        'cups': cups,
                        'target': {
                            'center': state.position,
                            'radius': 30  # Calculate actual radius
                        }
                    })
                
                # Record for learning
                await self.learner.record_frame(
                    [cup['position'] for cup in cups],
                    asyncio.get_event_loop().time()
                )
                
                # Update performance metrics
                await self.performance.record_tracking(state.confidence)
    
    async def _process_learning(self):
        """Process learning mode"""
        # Collect patterns and update models
        await self.learner.collect_patterns()
        
        # Adapt tracker parameters
        metrics = await self.performance.get_metrics()
        self.tracker.adapt_parameters(metrics)
        
        await asyncio.sleep(0.5)
    
    async def start_tracking(self, target_position: Tuple[int, int]):
        """Start tracking a specific cup"""
        logger.info(f"Starting tracking at position {target_position}")
        
        # Capture current frame
        frame = await self.system.capture_frame()
        if frame is None:
            return
        
        # Initialize tracker
        bbox = (target_position[0] - 30, target_position[1] - 30, 60, 60)
        self.tracker.initialize(frame, bbox)
        
        self.tracking_active = True
        self.current_mode = 'tracking'
        
        # Start pattern recording
        self.learner.start_pattern_recording()
        
    def stop_tracking(self):
        """Stop tracking"""
        logger.info("Stopping tracking")
        self.tracking_active = False
        self.current_mode = 'idle'
        
        # End pattern recording
        pattern = self.learner.end_pattern_recording('completed')
        if pattern:
            logger.info(f"Recorded pattern with complexity {pattern.complexity:.2f}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Cup Tracker Pro...")
        
        self.running = False
        
        # Stop components
        if self.overlay:
            await self.overlay.cleanup()
        
        if self.control_panel:
            await self.control_panel.cleanup()
        
        if self.performance:
            await self.performance.stop()
        
        if self.telemetry:
            await self.telemetry.stop()
        
        # Save models and data
        self.tracker.save_model(Path(self.config['model_path']))
        
        logger.info("Shutdown complete")
        
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Cup Tracker Pro')
    parser.add_argument('-c', '--config', type=Path, help='Configuration file path')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-overlay', action='store_true', help='Disable overlay')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run application
    app = CupTrackerPro(args.config)
    
    if args.no_overlay:
        app.config['wayland_overlay'] = False
    
    try:
        asyncio.run(app.initialize())
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        asyncio.run(app.shutdown())
    
    return 0

if __name__ == '__main__':
    sys.exit(main())