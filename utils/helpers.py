def generate_result_filename(config, iter_idx):
    """
    Generates a structured filename for simulation results.
    
    Format:
    result_Topology#{name}_N#{n}_T#{t_evol}_Int#{method}_Jr#{jr}_JrL#{jrl}_JrR#{jrr}_FL#{fl}_FR#{fr}_In#{input}_Data#{data}_Iter#{iter}.pkl
    """
    # Helper to sanitize values
    def fmt(val):
#        return str(val).replace('.', 'p')
        return str(val)
        
    # Extract Parameters
    # Assumes 'param_names' dict exists in config, or falls back to 'name' parsing if needed.
    # We will ensure 'param_names' is populated in 01_config.
    
    params = config.get('param_names', {})
    
    topology = config.get('param_names', {}).get('topology', "Ladder")
    N = fmt(config.get('N', 0))
    n_rails = fmt(config.get('n_rails', 2))
    t_evol = fmt(config.get('t_evol', 0))
    integrator = config.get('integration_method', 'Unknown')
    
    # Couplings
    # Check if Multi-Rail
    if 'J_rails' in params:
        jr_str = f"JRails#{params.get('J_rails')}_JRungs#{params.get('J_rungs')}"
    else:
        # Legacy Ladder support
        jr = params.get('J_rungs', 'Unk')
        jrl = params.get('J_rail_left', 'Unk')
        jrr = params.get('J_rail_right', 'Unk')
        jr_str = f"J_rung#{jr}_J_rail_left#{jrl}_J_rail_right#{jrr}"

    # Fields
    if 'field_rails' in params:
        f_str = f"Fields#{params.get('field_rails')}"
    else:
        fl = params.get('field_L', 'Unk')
        fr = params.get('field_R', 'Unk')
        f_str = f"field_rail_Left#{fl}_field_rail_Right#{fr}"
    
    # Input/Data
    inp_state = config.get('input_state_type', 'Unk')
    data_mode = config.get('data_input_type', 'Unk')
    
    # Disorder
    is_disordered = config.get('field_disorder', True)
    dis_str = "Dis" if is_disordered else "Uni"
    
    # Construct parts
    # Construct parts
    parts = [
        "result",
        f"Topology#{topology}",
        f"NRails#{n_rails}",
        f"LRail#{config.get('L', 'UNK')}",
        f"NTotal#{N}",
        f"T_evol#{t_evol}",
        f"Int#{integrator}",
        jr_str,
        f_str,
        f"InputState#{inp_state}",
        f"DataMode#{data_mode}",
        f"Disorder#{dis_str}",
        f"Iter#{iter_idx}"
    ]
    
    filename = "_".join(parts) + ".pkl"
    return filename
