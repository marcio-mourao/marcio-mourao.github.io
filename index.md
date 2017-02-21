var newLine, el, title, link;

var ToC =
  "<nav role='navigation' class='table-of-contents'>" +
    "<h2>On this page:</h2>" +
    "<ul>";
    
$("h2").each(function() {
  el = $(this);
  title = el.text();
  link = "#" + el.attr("id");
  
  newLine =
    "<li>" +
      "<a href='" + link + "'>" +
        title +
      "</a>" +
    "</li>";
    
  TOC += newLine;
});

ToC +=
   "</ul>" +
  "</nav>";


<h2 id="MATLAB">Introduction to MATLAB</h2>

### Part I

<a href="https://github.com/marcio-mourao/intro2MATLAB-1" target="_blank">Repository</a>

### Part II

<a href="https://github.com/marcio-mourao/intro2MATLAB-2" target="_blank">Repository</a>

<h2 id="SocialDataScience">Social Data Science</h2>

### Part I

<a href="https://github.com/marcio-mourao/socialDataScience-1" target="_blank">Repository</a>

### Part II

<a href="https://github.com/marcio-mourao/socialDataScience-2" target="_blank">Repository</a>
