#
# EDA shiny app
#
# data size: 1024MB
options(shiny.maxRequestSize = 1024*1024^2)
library(ggplot2)
library(shiny)
library(bs4Dash)
library(colourpicker)
library(shinyTime)
library(shinyWidgets)
library(thematic)
library(waiter)
library(DT)

thematic_shiny()

# color statuses
statusColors <- c("gray-dark","gray", "secondary","navy", "indigo", "purple",
                  "primary","lightblue","info", "success","olive","teal", "lime", "warning",
                  "orange", "danger", "fuchsia","maroon", "pink", "white")
# Function
Missing.Table <- function(data){
  output <- data.frame(type = sapply(data,class),
                       unique_value = sapply(data, function(x){if(!is.numeric(x)){length(unique(x))}
                         else{"---"}}),
                       num_missing  = sapply(data, function(x){sum(is.na(x))}),
                       missing_rate = sapply(data, function(x) round(mean(is.na(x))*100, 2)))
  rownames(output) <- names(data)
  colnames(output) <- c("type","# of unique value",
                        "# of missing value", "missing rate(%)")
  return(output)
}
Summary.Table <- function(data){
  numeric.col <- sapply(data, function(x){if (is.numeric(x)){TRUE}
    else{FALSE}})
  data.cont <- data[, numeric.col]
  output <- data.frame(min    = sapply(data.cont, function(x) min(x, na.rm = T)),
                       Q1     = sapply(data.cont, function(x) quantile(x, probs = c(0.25), na.rm = T)),
                       median = sapply(data.cont, function(x) median(x, na.rm = T)),
                       Q3     = sapply(data.cont, function(x) quantile(x, probs = c(0.75), na.rm = T)),
                       max    = sapply(data.cont, function(x) max(x, na.rm = T)))
  return(output)
}



# "Table Summary" tab
data.summary.tab <- tabItem(tabName = "DataSummary",
  fluidRow(
    box(title = "Import data", width = 3, status = "danger",
        solidHeader = FALSE, fileInput(inputId = "file", label = "Upload .csv file")),
    box(title = "Data Preview", width = 9, status = "primary", solidHeader = FALSE, 
        textOutput(outputId = "dim"), br(),
        DTOutput(outputId = "DataPreview", width = "100%"))
  ),
    box(title = "Missing value", width = 12, status = "secondary", solidHeader = FALSE, 
        DTOutput(outputId = "MissingPreview", width = "100%")),
    box(title = "Data Summary", width = 12, status = "info", solidHeader = FALSE,
        DTOutput(outputId = "DataSummary", width = "100%"))
)

# "Statistical Tests" tab
test.tab <- tabItem(tabName = "test",
  fluidRow(
    tabsetPanel(id = "testpanel", selected = "One group", vertical = FALSE,
      tabPanel(title = "One group", 
        tabsetPanel(id = "onegroup", selected = "One-sample t test", vertical = FALSE,
          tabPanel(title = "One-sample t test", 
                   selectInput(inputId = "OneSampleT.var1", label = "Select a Continuous Variable",
                               choices = NULL)))                 
        ),
      tabPanel(title = "Two groups", "This is two."),
      tabPanel(title = "Three or more groups", "This is three."))
  )
)

# "Plots" tab
plots.tab <- tabItem(tabName = "plots",
                     fluidRow())
# ui
ui <- dashboardPage(dark = FALSE, fullscreen = TRUE, scrollToTop = TRUE, 
  header = dashboardHeader(title = dashboardBrand(paste("EDA Shiny App", Sys.Date()), 
                                                  color = "primary")),
  sidebar = dashboardSidebar(fixed = TRUE, skin = "light", status = "primary",
                             id = "sidebar",
                  sidebarMenu(id = "current tab", flat = FALSE, compact = FALSE,
                              childIndent = TRUE,
                  menuItem(text = "Table Summary", tabName = "DataSummary"),
                  menuItem(text = "Statistical Tests", tabName = "test"),
                  menuItem(text = "Plots", tabName = "plots"))),
  body = dashboardBody(tabItems(data.summary.tab, test.tab, plots.tab))
)

server <- function(input, output, session){
  
  # Read input file
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })
  
  observe({
    req(data())
    numeric_vars <- names(data())[sapply(data(), is.numeric)]
    updateSelectInput(session, "OneSampleT.var1", choices = numeric_vars)
  })
  
  output$DataPreview <- renderDT({
    req(data())
    datatable(data(), options = list(pageLength = 5, scrollX = TRUE), 
              rownames = F)
  })
  output$dim <- renderText({
    paste("Dimension: ", dim(data())[1], "x", dim(data())[2])
  })
    
  output$MissingPreview <- renderDT({
    req(data())
    out <- Missing.Table(data())
    datatable(out, options = list(pageLength = 5, scrollX = TRUE), 
              rownames = T)
  })
  
  output$DataSummary <- renderDT({
    req(data())
    out <- Summary.Table(data())
    datatable(out, options = list(pageLength = 5, scrollX = TRUE), 
              rownames = T)
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
