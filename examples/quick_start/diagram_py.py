import trace

def trace_files(target_file):
    """
    Runs the target Python file and logs the filenames containing called functions.
    """
    # Only track function calls without printing line-by-line output
    tracer = trace.Trace(count=False, trace=False)
    tracer.run(f'exec(open("{target_file}").read())')

    # Retrieve the files from the tracer, getting only the ones that were executed
    results = tracer.results()
    files_called = set(results._files.keys())  # Collect unique files involved in function calls
    
    return files_called

def display_results(files_called):
    """
    Prints a unique list of files containing called functions.
    """
    print("Files Accessed:")
    for file in sorted(files_called):
        print(file)

def main():
    target_file = "examples/quick_start/nrms_ebnerd.py"  # Replace with your target script
    files_called = trace_files(target_file)
    display_results(files_called)

if __name__ == "__main__":
    main()
