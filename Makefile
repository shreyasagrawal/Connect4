CC := nvcc
CFLAGS := -g

all: final

clean:
	rm -f final

gpu: gpu.cu Makefile
	$(CC) $(CFLAGS) -o gpu gpu.cu

zip:
	@echo "Generating sudoku.zip file to submit to Gradescope..."
	@zip -q -r final.zip . -x .git/\* .vscode/\* .clang-format .gitignore final
	@echo "Done. Please upload sudoku.zip to Gradescope."

format:
	@echo "Reformatting source code."
	@clang-format -i --style=file $(wildcard *.c) $(wildcard *.h) $(wildcard *.cu)
	@echo "Done."

.PHONY: all clean zip format

