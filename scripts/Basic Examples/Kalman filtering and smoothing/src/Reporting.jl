module Reporting

export setup_report, append_to_report, generate_markdown_table

"""
    setup_report(output_dir)

Initialize the analysis report file. Returns the path to the report.
"""
function setup_report(output_dir)
    report_path = joinpath(output_dir, "analysis_report.txt")
    open(report_path, "w") do io
        println(io, "Kalman Filter Analysis Report")
        println(io, "=============================\n")
    end
    return report_path
end

"""
    append_to_report(path, text)

Append text to the analysis report.
"""
function append_to_report(path, text)
    open(path, "a") do io
        println(io, text)
    end
end

"""
    generate_markdown_table(headers::Vector{String}, rows::Vector{Vector{String}})

Generate a Markdown-formatted table string.
"""
function generate_markdown_table(headers::Vector{String}, rows::Vector{Vector{String}})
    # Calculate column widths
    widths = length.(headers)
    for row in rows
        widths = max.(widths, length.(row))
    end
    
    # Pad strings to width
    pad(s, w) = s * " "^(w - length(s))
    
    # helper to build row
    build_row(items) = "| " * join([pad(item, width) for (item, width) in zip(items, widths)], " | ") * " |"
    
    # Build header and separator
    header_row = build_row(headers)
    separator_row = "| " * join([repeat("-", width) for width in widths], " | ") * " |"
    
    # Build content rows
    content_rows = [build_row(row) for row in rows]
    
    return join([header_row, separator_row, content_rows...], "\n")
end

end # module
