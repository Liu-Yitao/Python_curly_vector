"""
CurlyVector: A third-party library for creating curly vector plots with matplotlib.

This library provides functionality to create vector field visualizations where
arrows follow the curvature of the field lines, with arrow lengths proportional
to vector magnitude.

Key Features:
- Field-following curved arrows
- Dynamic arrow sizing based on vector magnitude
- Cartopy integration for geographic projections
- Robust handling of missing data (NaN values)
- Customizable arrowhead styling

Author: Generated from curly_quivers notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator

# Optional Cartopy support
try:
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    GeoAxes = type(None)

class CurlyVector:
    """
    Creates individual curved arrows that follow vector field lines.
    
    This class implements the core functionality for generating a single
    curly vector arrow by integrating through the vector field to find
    the curved path from tail to head.
    """
    
    def __init__(self, x, y, u, v, interpolator, transform=None, scale=1.0, 
                 step_size=None, max_steps=1000, head_length=0.3, head_width=0.12, 
                 min_head_size_factor=2.0, max_head_size_factor=6.0, **kwargs):
        """
        Initialize a CurlyVector object.
        
        Parameters:
        -----------
        x, y : float
            Starting position coordinates
        u, v : float
            Vector components at starting position
        interpolator : callable
            Function that returns (u, v) at any (x, y) position
        transform : cartopy.crs, optional
            Source coordinate reference system for geographic data
        scale : float, default=1.0
            Scale factor for arrow length
        step_size : float, optional
            Integration step size (auto-calculated if None)
        max_steps : int, default=1000
            Maximum integration steps
        head_length : float, default=0.3
            Arrowhead length as fraction of arrow length
        head_width : float, default=0.15
            Arrowhead width as fraction of arrow length (reduced for sharper arrows)
        min_head_size_factor : float, default=2.0
            Minimum arrowhead size factor based on linewidth.
            min_head_length = linewidth * min_head_size_factor
            min_head_width = linewidth * min_head_size_factor
        max_head_size_factor : float, default=8.0
            Maximum arrowhead size factor based on linewidth.
            max_head_length = linewidth * max_head_size_factor
            max_head_width = linewidth * max_head_size_factor
        **kwargs : dict
            Additional styling arguments passed to FancyArrowPatch
        """
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.interpolator = interpolator
        self.transform = transform
        self.scale = scale
        self.step_size = step_size
        self.max_steps = max_steps
        self.head_length = head_length
        self.head_width = head_width
        self.min_head_size_factor = min_head_size_factor
        self.max_head_size_factor = max_head_size_factor
        self.kwargs = kwargs

    def _get_path_vertices(self):
        """
        Compute curved path vertices from tail to head via backward integration.
        
        Returns:
        --------
        list or None
            List of (x, y) coordinates from tail to head, or None if invalid
        """
        # Check for invalid vectors
        if np.isnan(self.u) or np.isnan(self.v):
            return None
            
        # Calculate total integration distance based on vector magnitude
        s0 = self.scale * np.hypot(self.u, self.v)
        if s0 < 1e-6:  # Skip very small vectors
            return None

        # Set integration step size
        step_size = min(s0 / 100.0, 0.01) if self.step_size is None else self.step_size
        
        # Start backward integration from head position
        points = [(self.x, self.y)]
        total_length = 0
        current = np.array([self.x, self.y])
        steps = 0
        
        # Backward integration to find tail position
        while total_length < s0 and steps < self.max_steps:
            try:
                # Get vector at current position
                uv = self.interpolator(np.array([[current[0], current[1]]]))
                u, v = uv[0]
                
                # Handle invalid interpolation results
                if np.isnan(u) or np.isnan(v):
                    break
                    
                speed = np.hypot(u, v)
                if speed < 1e-6:  # Handle zero vectors
                    break

                # Take step in backward direction
                step = min(step_size, s0 - total_length)
                direction = np.array([-u, -v]) / speed  # Backward direction
                current = current + step * direction
                total_length += step
                points.append(tuple(current))
                steps += 1
                
            except Exception:
                break
                
        # Need at least 2 points for a valid path
        if len(points) < 2:
            return None
            
        # Return path from tail to head
        return list(reversed(points))

    def get_arrow_patch(self, target_crs=None):
        """
        Create matplotlib arrow patch for rendering.
        
        Parameters:
        -----------
        target_crs : cartopy.crs, optional
            Target coordinate reference system for transformation
            
        Returns:
        --------
        FancyArrowPatch or None
            Matplotlib arrow patch ready for adding to axes
        """
        # Get path vertices
        verts = self._get_path_vertices()
        if verts is None or len(verts) < 2:
            return None

        # Create path codes for matplotlib
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        
        # Apply coordinate transformation if needed
        if self.transform is not None and target_crs is not None:
            transformed_verts = []
            for x, y in verts:
                try:
                    # Transform from source CRS to target CRS
                    transformed = target_crs.transform_point(x, y, self.transform)
                    transformed_verts.append(transformed)
                except Exception:
                    continue
                    
            if len(transformed_verts) < 2:
                return None
                
            path = Path(transformed_verts, codes)
        else:
            path = Path(verts, codes)
        
        # Calculate absolute arrowhead dimensions with min/max size limits based on linewidth
        arrow_length = self.scale * np.hypot(self.u, self.v)
        
        # Get linewidth from kwargs, default to 1.0 if not specified
        linewidth = self.kwargs.get('linewidth', 1.0)
        
        # Calculate minimum and maximum arrowhead sizes based on linewidth
        min_head_length = linewidth * self.min_head_size_factor
        min_head_width = linewidth * self.min_head_size_factor
        max_head_length = linewidth * self.max_head_size_factor
        max_head_width = linewidth * self.max_head_size_factor
        
        # Calculate arrowhead dimensions with both min and max size enforcement
        abs_head_length = np.clip(arrow_length * self.head_length, min_head_length, max_head_length)
        abs_head_width = np.clip(arrow_length * self.head_width, min_head_width, max_head_width)
        
        # Filter out custom parameters from kwargs before passing to FancyArrowPatch
        arrow_kwargs = {k: v for k, v in self.kwargs.items() 
                       if k not in ['min_head_size_factor', 'max_head_size_factor']}
        
        # Create arrow patch
        arrow = FancyArrowPatch(
            path=path,
            arrowstyle=f"->,head_length={abs_head_length},head_width={abs_head_width}",
            **arrow_kwargs
        )
        
        return arrow

def create_field_interpolator(X, Y, U, V):
    """
    Creates a continuous vector field interpolator from discrete grid data.
    
    This function handles the conversion of discrete grid data into a continuous
    field that can be sampled at any location for path integration.
    
    Parameters:
    -----------
    X, Y : array_like
        2D coordinate arrays (meshgrid format)
    U, V : array_like
        2D vector component arrays
        
    Returns:
    --------
    callable
        Interpolator function that returns (u, v) at any (x, y)
    """
    # Handle all-NaN case
    mask = ~np.isnan(U) & ~np.isnan(V)
    if not np.any(mask):
        return lambda points: np.zeros((len(points), 2))
    
    try:
        # Extract grid coordinates
        x_vals = np.unique(X[0, :])
        y_vals = np.unique(Y[:, 0])
        
        # Replace NaN values with zeros for interpolation
        U_safe = np.where(np.isnan(U), 0, U)
        V_safe = np.where(np.isnan(V), 0, V)
        
        # Reshape for RegularGridInterpolator
        U_grid = U_safe.reshape(len(y_vals), len(x_vals))
        V_grid = V_safe.reshape(len(y_vals), len(x_vals))
        
        # Combine U and V into single array
        UV = np.dstack((U_grid, V_grid))
        
        # Create interpolator (note: expects (y, x) coordinate order)
        interpolator = RegularGridInterpolator(
            (y_vals, x_vals), UV, bounds_error=False, fill_value=0
        )
        
        def interp_func(points):
            """
            Interpolate vector field at given points.
            
            Parameters:
            -----------
            points : array_like
                Array of shape (N, 2) with columns [x, y]
                
            Returns:
            --------
            array_like
                Array of shape (N, 2) with columns [u, v]
            """
            # RegularGridInterpolator expects (y, x) order
            points_array = np.array(points)
            if points_array.ndim == 1:
                points_array = points_array.reshape(1, -1)
            points_swapped = points_array[:, [1, 0]]  # Swap x,y to y,x
            return interpolator(points_swapped)
            
        return interp_func
        
    except Exception:
        # Fallback to returning zeros
        return lambda points: np.zeros((len(points), 2))

def curly_vector_plot(ax, X, Y, U, V, transform=None, scale=1.0, step_size=None, 
                      max_steps=1000, head_length=0.3, head_width=0.12, 
                      min_head_size_factor=2.0, max_head_size_factor=6.0, **kwargs):
    """
    Plot a field of curly vectors on matplotlib axes.
    
    This is the main plotting function that creates curved arrows for each
    valid grid point in the vector field. Works seamlessly with both regular
    matplotlib axes and Cartopy GeoAxes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Matplotlib axes object to plot on
    X, Y : array_like
        2D coordinate arrays (meshgrid format)
        For geographic data: X=longitude, Y=latitude
    U, V : array_like
        2D vector component arrays (same shape as X, Y)
    transform : cartopy.crs, optional
        Source coordinate reference system for geographic data
        Use ccrs.PlateCarree() for lat/lon data
    scale : float, default=1.0
        Scale factor for arrow lengths
    step_size : float, optional
        Integration step size (auto-calculated if None)
        For geographic data, use larger values (e.g., 0.5-2.0)
    max_steps : int, default=1000
        Maximum integration steps per arrow
    head_length : float, default=0.3
        Arrowhead length as fraction of arrow length
    head_width : float, default=0.15
        Arrowhead width as fraction of arrow length (reduced for sharper arrows)
    min_head_size_factor : float, default=2.0
        Minimum arrowhead size factor based on linewidth.
        Ensures arrowheads remain visible even for slow wind speeds.
        min_head_length = linewidth * min_head_size_factor
        min_head_width = linewidth * min_head_size_factor
    max_head_size_factor : float, default=8.0
        Maximum arrowhead size factor based on linewidth.
        Prevents arrowheads from becoming too large for fast wind speeds.
        max_head_length = linewidth * max_head_size_factor
        max_head_width = linewidth * max_head_size_factor
    **kwargs : dict
        Additional styling arguments (color, linewidth, alpha, etc.)
        
    Returns:
    --------
    list
        List of arrow patches added to the axes
        
    Examples:
    ---------
    # Basic usage with regular matplotlib
    fig, ax = plt.subplots()
    curly_vector_plot(ax, x_grid, y_grid, u_data, v_data)
    
    # Geographic usage with Cartopy
    import cartopy.crs as ccrs
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    curly_vector_plot(ax, lon_grid, lat_grid, u_wind, v_wind, 
                     transform=ccrs.PlateCarree(), scale=0.5)
    
    # With custom minimum arrowhead size for better visibility
    curly_vector_plot(ax, x_grid, y_grid, u_data, v_data,
                     linewidth=2.0, min_head_size_factor=1.5)
    
    # With both minimum and maximum arrowhead size limits and sharper arrows
    curly_vector_plot(ax, x_grid, y_grid, u_data, v_data,
                     linewidth=1.5, head_width=0.12, 
                     min_head_size_factor=2.0, max_head_size_factor=6.0)
    """
    # Create field interpolator
    interpolator = create_field_interpolator(X, Y, U, V)
    
    # Detect if this is a geographic plot
    is_geographic = CARTOPY_AVAILABLE and isinstance(ax, GeoAxes)
    target_crs = ax.projection if is_geographic else None
    
    # Auto-adjust parameters for geographic data
    if is_geographic and step_size is None:
        # Use larger step sizes for geographic data
        mean_spacing = np.mean([
            np.mean(np.diff(np.unique(X[0, :]))),
            np.mean(np.diff(np.unique(Y[:, 0])))
        ])
        step_size = mean_spacing * 0.1
    
    # Generate arrows for each grid point
    arrows = []
    
    # Filter out function-specific parameters from kwargs before passing to CurlyVector
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['min_head_size_factor', 'max_head_size_factor', 
                                   'head_length', 'head_width', 'step_size', 'max_steps']}
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            u, v = U[i, j], V[i, j]
            
            # Skip invalid vectors
            if np.isnan(u) or np.isnan(v):
                continue
            if np.hypot(u, v) < 1e-6:
                continue
                
            # Create curly vector
            vector = CurlyVector(
                x, y, u, v, interpolator, 
                transform=transform,
                scale=scale,
                step_size=step_size,
                max_steps=max_steps,
                head_length=head_length,
                head_width=head_width,
                min_head_size_factor=min_head_size_factor,
                max_head_size_factor=max_head_size_factor,
                **filtered_kwargs
            )
            
            # Get arrow patch
            arrow_patch = vector.get_arrow_patch(target_crs)
            if arrow_patch is None:
                continue
                
            # Set appropriate coordinate transform
            try:
                if is_geographic and transform is not None:
                    arrow_patch.set_transform(target_crs._as_mpl_transform(ax))
                else:
                    arrow_patch.set_transform(ax.transData)
                    
                ax.add_patch(arrow_patch)
                arrows.append(arrow_patch)
            except Exception:
                continue
    
    # Set axis limits for non-geographic plots
    if not is_geographic:
        ax.set_xlim(np.nanmin(X), np.nanmax(X))
        ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
        ax.set_aspect('equal')
    
    return arrows

def curly_vector_key(ax, X, Y, U, V, scale=1.0, key_length=1.0, label='1 unit', 
                     box=True, spacing=None, box_size=1.0, 
                     loc=None, loc_coordinate='axes', head_length=0.3, head_width=0.12,
                     min_head_size_factor=2.0, max_head_size_factor=6.0, **kwargs):
    """
    Add a straight legend key for curly vectors with optional background box.
    
    Creates a reference arrow to show the scale of the vector field.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object
    X, Y : array_like
        2D coordinate arrays (used for positioning when loc is None)
    U, V : array_like
        2D vector component arrays (not used but kept for consistency)
    scale : float, default=1.0
        Scale factor (should match main plot)
    key_length : float, default=1.0
        Length of the reference vector in data units
    label : str, default='1 unit'
        Text label for the key
    box : bool, default=True
        Whether to add a white background box
    spacing : float, optional
        Fixed spacing from plot edges in data units. If None, uses adaptive spacing.
        Only used when loc is None
    box_size : float, default=1.0
        Multiplier for the background box size (1.0 = default size)
    loc : tuple or list, optional
        Exact position coordinates [x, y] for the legend arrow.
        If None, uses default upper right positioning.
        Similar to matplotlib's quiverkey loc parameter.
    loc_coordinate : str, default='axes'
        Coordinate system for loc parameter:
        - 'axes': normalized axes coordinates (0-1)
        - 'data': data coordinates
        - 'figure': normalized figure coordinates (0-1)
    head_length : float, default=0.3
        Arrowhead length as fraction of arrow length (should match main plot)
    head_width : float, default=0.12
        Arrowhead width as fraction of arrow length (should match main plot)
    min_head_size_factor : float, default=2.0
        Minimum arrowhead size factor based on linewidth.
        Ensures arrowheads remain visible even for small legend arrows.
        min_head_length = linewidth * min_head_size_factor
        min_head_width = linewidth * min_head_size_factor
    max_head_size_factor : float, default=8.0
        Maximum arrowhead size factor based on linewidth.
        Prevents arrowheads from becoming too large for large legend arrows.
        max_head_length = linewidth * max_head_size_factor
        max_head_width = linewidth * max_head_size_factor
    **kwargs : dict
        Additional styling arguments (color, linewidth, etc.)
        
    Examples:
    ---------
    # Default positioning (upper right)
    curly_vector_key(ax, X, Y, U, V)
    
    # Exact positioning in axes coordinates
    curly_vector_key(ax, X, Y, U, V, loc=[0.85, 0.9], loc_coordinate='axes')
    
    # Exact positioning in data coordinates
    curly_vector_key(ax, X, Y, U, V, loc=[100, 50], loc_coordinate='data')
    
    # With custom minimum arrowhead size
    curly_vector_key(ax, X, Y, U, V, linewidth=1.5, min_head_size_factor=1.0)
    
    # With both minimum and maximum arrowhead size limits
    curly_vector_key(ax, X, Y, U, V, linewidth=1.5, 
                     head_length=0.3, head_width=0.12,
                     min_head_size_factor=2.0, max_head_size_factor=6.0)
    
    # Ensuring consistency with main plot parameters
    curly_vector_plot(ax, X, Y, U, V, head_length=0.25, head_width=0.10, ...)
    curly_vector_key(ax, X, Y, U, V, head_length=0.25, head_width=0.10, ...)
    """
    
    # Calculate key position
    if loc is not None:
        # Use exact positioning with loc parameter
        if loc_coordinate == 'axes':
            # Convert normalized axes coordinates (0-1) to data coordinates
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x0 = xlim[0] + loc[0] * (xlim[1] - xlim[0])
            y0 = ylim[0] + loc[1] * (ylim[1] - ylim[0])
        elif loc_coordinate == 'data':
            # Use data coordinates directly
            x0, y0 = loc[0], loc[1]
        elif loc_coordinate == 'figure':
            # Convert figure coordinates to data coordinates
            fig = ax.figure
            # Transform from figure coordinates to display coordinates
            display_coords = fig.transFigure.transform(loc)
            # Transform from display coordinates to data coordinates
            data_coords = ax.transData.inverted().transform(display_coords)
            x0, y0 = data_coords[0], data_coords[1]
        else:
            raise ValueError(f"loc_coordinate must be 'axes', 'data', or 'figure', got '{loc_coordinate}'")
    else:
        # Use automatic positioning with position parameter (original behavior)
        x_range = np.nanmax(X) - np.nanmin(X)
        y_range = np.nanmax(Y) - np.nanmin(Y)
        
        # Use fixed spacing if provided, otherwise use adaptive spacing
        if spacing is None:
            # Adaptive spacing based on data range (original behavior)
            x_margin = 0.25 * x_range
            y_margin = 0.15 * y_range
            x_margin_small = 0.05 * x_range
            y_margin_small = 0.05 * y_range
        else:
            # Fixed spacing in data units
            x_margin = spacing
            y_margin = spacing
            x_margin_small = spacing * 0.5
            y_margin_small = spacing * 0.5
        
        # Default to upper right positioning (most common case)
        x0 = np.nanmax(X) - x_margin
        y0 = np.nanmax(Y) - y_margin
    
    # Calculate arrow length
    dx = key_length * scale
    
    # Add background box if requested
    if box:
        from matplotlib.patches import Rectangle
        
        # Calculate base box dimensions
        base_box_width = dx * 1.6
        if loc is not None:
            # For exact positioning, use reasonable defaults
            if loc_coordinate == 'data':
                # Try to estimate appropriate size based on data range
                try:
                    x_range = np.nanmax(X) - np.nanmin(X)
                    y_range = np.nanmax(Y) - np.nanmin(Y)
                    base_box_height = 0.08 * y_range
                except:
                    # Fallback if X,Y are not available
                    base_box_height = dx * 0.5
            else:
                # For axes/figure coordinates, use relative sizing
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                y_range = ylim[1] - ylim[0]
                base_box_height = 0.08 * y_range
        elif spacing is None:
            y_range = np.nanmax(Y) - np.nanmin(Y)
            base_box_height = 0.08 * y_range
        else:
            base_box_height = spacing * 0.8  # Consistent box height
        
        # Apply box_size multiplier to adjust box dimensions
        box_width = base_box_width * box_size
        box_height = base_box_height * box_size
        
        # Adjust box position to keep it centered around the arrow
        box_x = x0 - dx * 0.3 - (box_width - base_box_width) * 0.5
        box_y = y0 - box_height * 0.3
        
        rect = Rectangle((box_x, box_y), box_width, box_height,
                        facecolor='white', edgecolor='black', linewidth=0.8,
                        alpha=0.9, zorder=100)
        ax.add_patch(rect)
    
    # Create straight arrow using FancyArrowPatch (same as curly vectors)
    # This ensures perfect consistency in appearance and styling
    
    # Calculate absolute arrowhead dimensions with min/max size limits based on linewidth
    arrow_length = dx
    
    # Get linewidth from kwargs, default to 1.0 if not specified
    linewidth = kwargs.get('linewidth', 1.0)
    
    # Calculate minimum and maximum arrowhead sizes based on linewidth
    min_head_length = linewidth * min_head_size_factor
    min_head_width = linewidth * min_head_size_factor
    max_head_length = linewidth * max_head_size_factor
    max_head_width = linewidth * max_head_size_factor
    
    # Calculate arrowhead dimensions with both min and max size enforcement
    # Use the provided head_length and head_width parameters for consistency
    abs_head_length = np.clip(arrow_length * head_length, min_head_length, max_head_length)
    abs_head_width = np.clip(arrow_length * head_width, min_head_width, max_head_width)
    
    # Create straight path from start to end
    straight_path = Path([(x0, y0), (x0 + dx, y0)], 
                        [Path.MOVETO, Path.LINETO])
    
    # Filter out custom parameters from kwargs before passing to FancyArrowPatch
    arrow_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['min_head_size_factor', 'max_head_size_factor']}
    
    # Create FancyArrowPatch with same styling as curly vectors
    arrow_patch = FancyArrowPatch(
        path=straight_path,
        arrowstyle=f"->,head_length={abs_head_length},head_width={abs_head_width}",
        zorder=101,
        **arrow_kwargs
    )
    
    # Add the arrow patch to the axes
    ax.add_patch(arrow_patch)
    
    # Add label with appropriate spacing
    if loc is not None:
        # For exact positioning, use reasonable text offset
        if loc_coordinate == 'data':
            try:
                y_range = np.nanmax(Y) - np.nanmin(Y)
                text_y_offset = 0.03 * y_range
            except:
                # Fallback if Y is not available
                text_y_offset = dx * 0.2
        else:
            # For axes/figure coordinates, use relative sizing
            ylim = ax.get_ylim()
            y_range = ylim[1] - ylim[0]
            text_y_offset = 0.03 * y_range
    elif spacing is None:
        y_range = np.nanmax(Y) - np.nanmin(Y)
        text_y_offset = 0.03 * y_range
    else:
        text_y_offset = spacing * 0.3  # Consistent text offset
        
    ax.text(x0 + dx/2, y0 + text_y_offset, label, 
            ha='center', va='bottom', fontsize=10, zorder=102,
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8) if not box else None)

    # Handle backward compatibility for deprecated parameters
    if 'key_pos' in kwargs:
        import warnings
        warnings.warn("'key_pos' parameter is deprecated and no longer supported. Use 'loc' parameter for exact positioning.", 
                     DeprecationWarning, stacklevel=2)
        kwargs.pop('key_pos')
    
    if 'position' in kwargs:
        import warnings
        warnings.warn("'position' parameter is deprecated and no longer supported. Use 'loc' parameter for exact positioning.", 
                     DeprecationWarning, stacklevel=2)
        kwargs.pop('position')
    
    if 'position_text' in kwargs:
        import warnings
        warnings.warn("'position_text' parameter is deprecated and no longer supported. Use 'loc' parameter for exact positioning.", 
                     DeprecationWarning, stacklevel=2)
        kwargs.pop('position_text')

