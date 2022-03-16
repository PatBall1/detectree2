# Get general jupyter extensions
pip install jupyter
pip install jupyter_contrib_nbextensions
jupyter nbextensions_configurator enable --user
jupyter contrib nbextension install --user

# Get jupyter black
jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
jupyter nbextension enable jupyter-black-master/jupyter-black

# Activate line number toggeling
jupyter nbextension enable toggle_all_line_numbers/main
# Activate execution timing
jupyter nbextension enable execute_time/ExecuteTime
# Activate codefolding
jupyter nbextension enable codefolding/main
# Activate collapsible headings
jupyter nbextension enable collapsible_headings/main
# Activate highlighting selected word
jupyter nbextension enable highlight_selected_word/main
# Activate move selected cells
jupyter nbextension enable move_selected_cells/main
# Activate scratchpad
jupyter nbextension enable scratchpad/main
# Activate autosave
jupyter nbextension enable autosavetime/main
# Activate variable inspector
jupyter nbextension enable varInspector/main
# Activate cell freezing
jupyter nbextension enable freeze/main
# Activate hide input
jupyter nbextension enable hide_input_all/main