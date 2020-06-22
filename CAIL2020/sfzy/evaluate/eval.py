from pyrouge import Rouge155
r = Rouge155()

r.system_dir = 'system_summaries'
r.model_dir = 'model_summaries'
r.system_filename_pattern = 'text.(\d+).txt'
r.model_filename_pattern = 'text.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
