import pypandoc

# 输入和输出文件名以及模板
input_filename = 'storybook/movie.md'
output_filename = 'storybook/movie.pdf'
template = 'eisvogel'

# 设置额外的参数
extra_args = ['--template', template, '--listings']

# 使用pypandoc进行转换
output = pypandoc.convert_file(
    input_filename,
    'pdf',
    outputfile=output_filename,
    extra_args=extra_args
)

print(f"已将 {input_filename} 转换为 {output_filename} 使用模板 {template}")
