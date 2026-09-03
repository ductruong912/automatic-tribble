"""
Advanced preprocessing module for scanned documents before OCR.
Implements various image enhancement techniques to improve OCR accuracy.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io

logger = logging.getLogger(__name__)


class PreprocessingMethod(Enum):
    """Available preprocessing methods."""
    DESKEW = "deskew"
    DENOISE = "denoise"
    BINARIZE = "binarize"
    CONTRAST_ENHANCE = "contrast_enhance"
    SHARPEN = "sharpen"
    CROP_MARGINS = "crop_margins"
    ROTATE = "rotate"
    SCALE = "scale"
    REMOVE_ARTIFACTS = "remove_artifacts"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations."""
    # Deskewing
    deskew_enabled: bool = True
    deskew_threshold: float = 0.5  # degrees
    
    # Denoising
    denoise_enabled: bool = True
    denoise_method: str = "gaussian"  # gaussian, bilateral, median
    
    # Binarization
    binarize_enabled: bool = True
    binarize_method: str = "otsu"  # otsu, adaptive, threshold
    
    # Contrast enhancement
    contrast_enabled: bool = True
    contrast_factor: float = 1.2
    
    # Sharpening
    sharpen_enabled: bool = True
    sharpen_factor: float = 1.5
    
    # Cropping
    crop_margins_enabled: bool = True
    margin_threshold: int = 10
    
    # Scaling
    scale_enabled: bool = True
    target_dpi: int = 300
    max_scale_factor: float = 3.0
    
    # Artifact removal
    remove_artifacts_enabled: bool = True
    artifact_size_threshold: int = 50


class DocumentPreprocessor:
    """
    Advanced document preprocessing for OCR optimization.
    Implements multiple enhancement techniques to improve scan quality.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig()
        logger.info("DocumentPreprocessor initialized with config")
    
    def preprocess_image(self, image: np.ndarray, methods: Optional[List[PreprocessingMethod]] = None) -> np.ndarray:
        """
        Apply preprocessing methods to an image.
        
        Args:
            image: Input image as numpy array
            methods: List of methods to apply. Uses all enabled methods if None.
            
        Returns:
            Preprocessed image as numpy array
        """
        if methods is None:
            methods = self._get_enabled_methods()
        
        processed_image = image.copy()
        
        for method in methods:
            try:
                processed_image = self._apply_method(processed_image, method)
                logger.debug(f"Applied preprocessing method: {method.value}")
            except Exception as e:
                logger.warning(f"Failed to apply {method.value}: {e}")
                continue
        
        return processed_image
    
    def _get_enabled_methods(self) -> List[PreprocessingMethod]:
        """Get list of enabled preprocessing methods based on config."""
        methods = []
        
        if self.config.deskew_enabled:
            methods.append(PreprocessingMethod.DESKEW)
        if self.config.denoise_enabled:
            methods.append(PreprocessingMethod.DENOISE)
        if self.config.binarize_enabled:
            methods.append(PreprocessingMethod.BINARIZE)
        if self.config.contrast_enabled:
            methods.append(PreprocessingMethod.CONTRAST_ENHANCE)
        if self.config.sharpen_enabled:
            methods.append(PreprocessingMethod.SHARPEN)
        if self.config.crop_margins_enabled:
            methods.append(PreprocessingMethod.CROP_MARGINS)
        if self.config.scale_enabled:
            methods.append(PreprocessingMethod.SCALE)
        if self.config.remove_artifacts_enabled:
            methods.append(PreprocessingMethod.REMOVE_ARTIFACTS)
        
        return methods
    
    def _apply_method(self, image: np.ndarray, method: PreprocessingMethod) -> np.ndarray:
        """Apply a specific preprocessing method."""
        if method == PreprocessingMethod.DESKEW:
            return self._deskew_image(image)
        elif method == PreprocessingMethod.DENOISE:
            return self._denoise_image(image)
        elif method == PreprocessingMethod.BINARIZE:
            return self._binarize_image(image)
        elif method == PreprocessingMethod.CONTRAST_ENHANCE:
            return self._enhance_contrast(image)
        elif method == PreprocessingMethod.SHARPEN:
            return self._sharpen_image(image)
        elif method == PreprocessingMethod.CROP_MARGINS:
            return self._crop_margins(image)
        elif method == PreprocessingMethod.ROTATE:
            return self._rotate_image(image)
        elif method == PreprocessingMethod.SCALE:
            return self._scale_image(image)
        elif method == PreprocessingMethod.REMOVE_ARTIFACTS:
            return self._remove_artifacts(image)
        else:
            return image
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in scanned documents using Hough line detection.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) == 0:
                return image
            
            # Calculate angles from detected lines
            angles = []
            for line in lines[:min(20, len(lines))]:  # Limit for efficiency
                rho, theta = line[0]
                angle = theta - np.pi/2
                angles.append(angle)
            
            if not angles:
                return image
            
            # Calculate median angle
            median_angle = np.median(angles)
            
            # Only correct if angle is significant
            if abs(median_angle) > np.radians(self.config.deskew_threshold):
                # Rotate image to correct skew
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(median_angle), 1.0)
                
                # Calculate new dimensions to avoid cropping
                cos_val = abs(rotation_matrix[0, 0])
                sin_val = abs(rotation_matrix[0, 1])
                new_w = int((h * sin_val) + (w * cos_val))
                new_h = int((h * cos_val) + (w * sin_val))
                
                # Adjust rotation matrix for new center
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
                logger.debug(f"Deskewed image by {np.degrees(median_angle):.2f} degrees")
            
            return image
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from scanned documents.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if self.config.denoise_method == "gaussian":
                denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            elif self.config.denoise_method == "bilateral":
                denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            elif self.config.denoise_method == "median":
                denoised = cv2.medianBlur(gray, 3)
            else:
                denoised = gray
            
            # Convert back to original format
            if len(image.shape) == 3:
                return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            else:
                return denoised
                
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to binary (black and white) for better OCR.
        
        Args:
            image: Input image
            
        Returns:
            Binarized image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if self.config.binarize_method == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self.config.binarize_method == "adaptive":
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            elif self.config.binarize_method == "threshold":
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = gray
            
            # Convert back to original format
            if len(image.shape) == 3:
                return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            else:
                return binary
                
        except Exception as e:
            logger.warning(f"Binarization failed: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge channels and convert back to BGR
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                return enhanced
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
                return enhanced
                
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image to improve text clarity.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to control sharpening strength
            result = cv2.addWeighted(image, 1 - self.config.sharpen_factor, 
                                   sharpened, self.config.sharpen_factor, 0)
            
            return result
            
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image
    
    def _crop_margins(self, image: np.ndarray) -> np.ndarray:
        """
        Remove excessive margins from scanned documents.
        
        Args:
            image: Input image
            
        Returns:
            Cropped image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Find non-white regions
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Get bounding box of all contours
            x_min, y_min, x_max, y_max = image.shape[1], image.shape[0], 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Add small margin
            margin = self.config.margin_threshold
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(image.shape[1], x_max + margin)
            y_max = min(image.shape[0], y_max + margin)
            
            # Crop image
            cropped = image[y_min:y_max, x_min:x_max]
            
            logger.debug(f"Cropped margins: {image.shape} -> {cropped.shape}")
            return cropped
            
        except Exception as e:
            logger.warning(f"Margin cropping failed: {e}")
            return image
    
    def _rotate_image(self, image: np.ndarray, angle: float = 0) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        try:
            if abs(angle) < 0.1:  # Skip very small rotations
                return image
            
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Calculate new dimensions
            cos_val = abs(np.cos(np.radians(angle)))
            sin_val = abs(np.sin(np.radians(angle)))
            new_w = int((h * sin_val) + (w * cos_val))
            new_h = int((h * cos_val) + (w * sin_val))
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # Rotate image
            rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            logger.warning(f"Rotation failed: {e}")
            return image
    
    def _scale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Scale image to optimal resolution for OCR.
        
        Args:
            image: Input image
            
        Returns:
            Scaled image
        """
        try:
            height, width = image.shape[:2]
            
            # Calculate scale factor based on target DPI
            # Assuming input is around 72 DPI
            scale_factor = min(self.config.target_dpi / 72, self.config.max_scale_factor)
            
            if scale_factor > 1.1:  # Only scale if significant improvement
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                scaled = cv2.resize(image, (new_width, new_height), 
                                  interpolation=cv2.INTER_CUBIC)
                
                logger.debug(f"Scaled image by factor {scale_factor:.2f}")
                return scaled
            
            return image
            
        except Exception as e:
            logger.warning(f"Scaling failed: {e}")
            return image
    
    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small artifacts and noise spots.
        
        Args:
            image: Input image
            
        Returns:
            Cleaned image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Find contours
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask for artifacts
            mask = np.ones(gray.shape, dtype=np.uint8) * 255
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.artifact_size_threshold:
                    cv2.fillPoly(mask, [contour], 0)
            
            # Apply mask to remove artifacts
            cleaned = cv2.bitwise_and(gray, mask)
            
            # Convert back to original format
            if len(image.shape) == 3:
                return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            else:
                return cleaned
                
        except Exception as e:
            logger.warning(f"Artifact removal failed: {e}")
            return image
    
    def preprocess_pdf_page(self, page_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single PDF page with optimal settings for OCR.
        
        Args:
            page_image: PDF page as numpy array
            
        Returns:
            Preprocessed page image
        """
        return self.preprocess_image(page_image)
    
    def get_preprocessing_stats(self, original_image: np.ndarray, 
                              processed_image: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the preprocessing applied.
        
        Args:
            original_image: Original image
            processed_image: Processed image
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            "original_shape": original_image.shape,
            "processed_shape": processed_image.shape,
            "size_change": {
                "width": processed_image.shape[1] / original_image.shape[1],
                "height": processed_image.shape[0] / original_image.shape[0]
            },
            "methods_applied": [method.value for method in self._get_enabled_methods()],
            "config": {
                "deskew_enabled": self.config.deskew_enabled,
                "denoise_enabled": self.config.denoise_enabled,
                "binarize_enabled": self.config.binarize_enabled,
                "contrast_enabled": self.config.contrast_enabled,
                "sharpen_enabled": self.config.sharpen_enabled,
                "crop_margins_enabled": self.config.crop_margins_enabled,
                "scale_enabled": self.config.scale_enabled,
                "remove_artifacts_enabled": self.config.remove_artifacts_enabled
            }
        }
        
        return stats


# Convenience functions for common preprocessing tasks
def create_ocr_optimized_config() -> PreprocessingConfig:
    """Create a configuration optimized for OCR processing."""
    return PreprocessingConfig(
        deskew_enabled=True,
        deskew_threshold=0.5,
        denoise_enabled=True,
        denoise_method="bilateral",
        binarize_enabled=True,
        binarize_method="otsu",
        contrast_enabled=True,
        contrast_factor=1.2,
        sharpen_enabled=True,
        sharpen_factor=1.5,
        crop_margins_enabled=True,
        margin_threshold=10,
        scale_enabled=True,
        target_dpi=300,
        max_scale_factor=2.0,
        remove_artifacts_enabled=True,
        artifact_size_threshold=50
    )


def create_fast_config() -> PreprocessingConfig:
    """Create a configuration optimized for speed."""
    return PreprocessingConfig(
        deskew_enabled=False,
        denoise_enabled=True,
        denoise_method="gaussian",
        binarize_enabled=True,
        binarize_method="otsu",
        contrast_enabled=False,
        sharpen_enabled=False,
        crop_margins_enabled=False,
        scale_enabled=True,
        target_dpi=200,
        max_scale_factor=1.5,
        remove_artifacts_enabled=False
    )


def create_high_quality_config() -> PreprocessingConfig:
    """Create a configuration optimized for maximum quality."""
    return PreprocessingConfig(
        deskew_enabled=True,
        deskew_threshold=0.2,
        denoise_enabled=True,
        denoise_method="bilateral",
        binarize_enabled=True,
        binarize_method="adaptive",
        contrast_enabled=True,
        contrast_factor=1.5,
        sharpen_enabled=True,
        sharpen_factor=2.0,
        crop_margins_enabled=True,
        margin_threshold=5,
        scale_enabled=True,
        target_dpi=400,
        max_scale_factor=3.0,
        remove_artifacts_enabled=True,
        artifact_size_threshold=30
    )





