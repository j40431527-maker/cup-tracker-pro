# src/ui/wayland_overlay.py

import pywayland
from pywayland.server import Display, Client
from pywayland.protocol.wayland import WlCompositor, WlSubcompositor
from pywayland.protocol.viewporter import Viewporter
import ctypes
import numpy as np
from PIL import Image
import logging
import asyncio
from typing import Optional, Tuple, List
import struct

logger = logging.getLogger(__name__)

class WaylandOverlay:
    """High-performance Wayland overlay using direct compositor integration"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.display = None
        self.surface = None
        self.subsurface = None
        self.buffer = None
        self.shm_pool = None
        self.buffer_size = width * height * 4  # RGBA
        self.buffer_data = None
        
    async def initialize(self):
        """Initialize Wayland connection and create overlay surface"""
        # Connect to Wayland display
        self.display = pywayland.Client()
        self.display.connect()
        
        # Get globals
        registry = self.display.get_registry()
        
        await asyncio.sleep(0.1)  # Wait for registry events
        
        # Create surface
        self.surface = self.display.compositor.create_surface()
        
        # Create shared memory pool
        fd = self._create_shm_file(self.buffer_size)
        self.shm_pool = self.display.shm.create_pool(fd, self.buffer_size)
        
        # Create buffer
        self.buffer = self.shm_pool.create_buffer(0, self.width, self.height,
                                                  self.width * 4,
                                                  pywayland.ShmFormat.argb8888.value)
        
        # Map buffer memory
        self.buffer_data = ctypes.create_string_buffer(self.buffer_size)
        
        # Set up subsurface (for overlay)
        if hasattr(self.display, 'subcompositor'):
            parent = self._get_top_level_surface()
            self.subsurface = self.display.subcompositor.get_subsurface(self.surface, parent)
            self.subsurface.set_position(0, 0)
            self.subsurface.set_desync()
        
        # Set surface properties
        self.surface.set_buffer_scale(1)
        self.surface.set_buffer_transform(0)
        
        # Commit surface
        self.surface.damage(0, 0, self.width, self.height)
        self.surface.attach(self.buffer, 0, 0)
        self.surface.commit()
        
        logger.info("Wayland overlay initialized")
        
    def update_overlay(self, drawing_data: dict):
        """Update overlay with new drawing data"""
        if not self.buffer_data:
            return
            
        # Create RGBA image
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw cups
        if 'cups' in drawing_data:
            for cup in drawing_data['cups']:
                x, y = cup['center']
                r = cup['radius']
                
                # Draw cup outline
                draw.ellipse([x-r, y-r, x+r, y+r], 
                            outline=(0, 255, 0, 255), width=2)
        
        # Draw target
        if 'target' in drawing_data:
            x, y = drawing_data['target']['center']
            r = drawing_data['target']['radius']
            
            # Draw target highlight
            draw.ellipse([x-r-5, y-r-5, x+r+5, y+r+5],
                        outline=(255, 0, 0, 255), width=4)
            
            # Draw crosshair
            draw.line([x-20, y, x+20, y], fill=(255, 0, 0, 255), width=2)
            draw.line([x, y-20, x, y+20], fill=(255, 0, 0, 255), width=2)
            
            # Draw text
            draw.text((x-40, y-r-30), "🎯 TARGET", 
                     fill=(255, 0, 0, 255))
        
        # Convert to raw RGBA data
        img_data = img.tobytes('raw', 'RGBA')
        
        # Copy to shared memory
        ctypes.memmove(self.buffer_data, img_data, self.buffer_size)
        
        # Damage and commit
        self.surface.damage(0, 0, self.width, self.height)
        self.surface.commit()
        
        # Dispatch events
        self.display.dispatch()
        
    def _create_shm_file(self, size: int) -> int:
        """Create shared memory file for buffer"""
        import os
        import tempfile
        
        fd, path = tempfile.mkstemp()
        os.unlink(path)
        
        # Set size
        os.ftruncate(fd, size)
        
        return fd
    
    def _get_top_level_surface(self) -> Optional[pywayland.Surface]:
        """Get the top-level surface (game window)"""
        # This would need proper window detection
        # For now, return a dummy surface
        return self.display.compositor.create_surface()
    
    async def cleanup(self):
        """Clean up Wayland resources"""
        if self.surface:
            self.surface.destroy()
        if self.shm_pool:
            self.shm_pool.destroy()
        if self.display:
            self.display.disconnect()
        logger.info("Wayland overlay cleaned up")