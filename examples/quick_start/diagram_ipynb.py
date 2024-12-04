import sys
import os
from nbconvert import PythonExporter
from nbformat import read


class FileTrace:
    def __init__(self):
        self.visited_files = set()

    def trace_calls(self, frame, event, arg):
        """
        Function that tracks calls and adds file names to the set.
        """
        if event != "call":
            return self.trace_calls  # Only interested in "call" events
        
        code = frame.f_code
        filename = code.co_filename

        # Filter built-in or standard library modules
        if not filename.startswith("<") and "site-packages" not in filename:
            self.visited_files.add(filename)
        return self.trace_calls

    def start_tracing(self):
        """
        Start tracing by setting the custom trace function.
        """
        sys.settrace(self.trace_calls)

    def stop_tracing(self):
        """
        Stop tracing by disabling the trace function.
        """
        sys.settrace(None)


def convert_notebook_to_script(notebook_file):
    """
    Converts a Jupyter Notebook (.ipynb) to a Python script (.py).
    Returns the path of the temporary script.
    """
    with open(notebook_file, "r", encoding="utf-8") as nb_file:
        notebook_content = read(nb_file, as_version=4)
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(notebook_content)
    
    temp_script = notebook_file.replace(".ipynb", "_temp.py")
    with open(temp_script, "w", encoding="utf-8") as script_file:
        script_file.write(script)
    
    return temp_script


def main():
    notebook_file = "examples/quick_start/nrms_ebnerd.ipynb"  # Replace with your notebook
    temp_script = convert_notebook_to_script(notebook_file)
    tracer = FileTrace()

    try:
        tracer.start_tracing()
        exec(open(temp_script).read())  # Execute the temporary script
    finally:
        tracer.stop_tracing()
        if os.path.exists(temp_script):
            os.remove(temp_script)

    print("\nUnique Files Accessed During Execution:")
    for file in sorted(tracer.visited_files):
        print(file)


if __name__ == "__main__":
    main()

