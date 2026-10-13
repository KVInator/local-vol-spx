-- Allocate widths for the report's result tables before LaTeX typesetting.
function Table(table)
  local widths = {
    [5] = {0.20, 0.18, 0.18, 0.15, 0.29},
    [6] = {0.11, 0.19, 0.11, 0.15, 0.14, 0.30},
  }
  local chosen = widths[#table.colspecs]
  if chosen then
    for index, width in ipairs(chosen) do
      table.colspecs[index][2] = width
    end
  end
  if #table.colspecs == 5 then
    return {pandoc.RawBlock("latex", "\\Needspace{12\\baselineskip}"), table}
  end
  return table
end

-- Keep each execution-case heading and its complete result table together.
function Header(header)
  local title = pandoc.utils.stringify(header.content)
  if header.level == 2 and (
    title == "Midpoint execution" or
    title == "Option bid–ask execution" or
    title == "Option bid–ask plus hedge fee"
  ) then
    return {pandoc.RawBlock("latex", "\\clearpage"), header}
  end
end
