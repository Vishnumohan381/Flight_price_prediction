

  
document.getElementById("flightForm").addEventListener("submit", function(event) {
    event.preventDefault(); 

    
    const departure = document.getElementById("departure").value;
    const destination = document.getElementById("destination").value;
    const airline = document.getElementById("airline").value;
    const flightClass = document.getElementById("flightClass").value;
    const departureDate = document.getElementById("departureDate").value;
    
    
    if (!departure || !destination || !flightClass || !departureDate) {
        alert("Please fill in all the required fields.");
        return;
    }

     



   
    let price = simulatePricePrediction(flightClass);

    
    if (airline === "IndiGo") {
        price += 200;  
    } else if (airline === "Air India") {
        price += 400;  
    }

   
    document.getElementById('predict').textContent = `₹${price.toFixed(1)}`;
    


    
    document.getElementById('predictionResult').classList.remove('hidden');
    
   
});


function simulatePricePrediction(flightClass) {
    const basePrices = {
        "Second Class": 5000,
        "economy Class": 15000,
        "First Class": 30000
    };



    function simulatePricePrediction(departureCity, destinationCity, flightClass) {
        // Base price by flight class
        const basePrices = {
            "Second Class": 5000,
            "economy Class": 15000,
            "First Class": 30000
        };
    
        // Base price based on flight class (default to Second Class if invalid class)
        let basePrice = basePrices[flightClass] || 5000;
    
        // City price adjustment factors
        const cityPriceAdjustments = {
            "Kochi": 0.8,   // Kochi is cheaper by 20%
            "Mumbai": 1.2,   // Mumbai is 20% more expensive
            "Delhi": 1.1,    // Delhi is 10% more expensive
            "Chennai": 1.0,  // Chennai is neutral
            "Bangalore": 1.1, // Bangalore is 10% more expensive
            "Hyderabad": 1.0, // Hyderabad is neutral
            "Kolkata": 0.9    // Kolkata is 10% cheaper
        };
    
        // Get city-based adjustment multiplier for departure and destination
        let departureAdjustment = cityPriceAdjustments[departureCity] || 1.0; // Default to 1 if city not found
        let destinationAdjustment = cityPriceAdjustments[destinationCity] || 1.0; // Default to 1 if city not found
    
        // Debugging output
        console.log(`Base price for ${flightClass}: ${basePrice}`);
        console.log(`Departure city adjustment for ${departureCity}: ${departureAdjustment}`);
        console.log(`Destination city adjustment for ${destinationCity}: ${destinationAdjustment}`);
    
        // Adjust the price based on city factors (both departure and destination)
        let adjustedPrice = basePrice * departureAdjustment * destinationAdjustment;
    
        // Return the final adjusted price
        return Math.round(adjustedPrice);
    }
    
    // Example usage
    console.log(simulatePricePrediction("Mumbai", "Delhi", "First Class"));
    

    let basePrice = basePrices[flightClass] || 5000;

  
    let priceFluctuation = localStorage.getItem('priceFluctuation');

   
    if (priceFluctuation === null) {
        priceFluctuation = Math.random() * 5000;
        localStorage.setItem('priceFluctuation', priceFluctuation);
    } else {
        
        priceFluctuation = parseFloat(priceFluctuation);
    }

    return Math.round(basePrice + priceFluctuation);
}


function calculateDistance(departure, destination) {
    const cityDistances = {
        "Chennai": { "Bangalore": 300, "Delhi": 1700, "Mumbai": 1300 },
        "Bangalore": { "Chennai": 300, "Delhi": 2000, "Mumbai": 980 },
        "Delhi": { "Chennai": 1700, "Bangalore": 2000, "Mumbai": 1400 },
        "Mumbai": { "Chennai": 1300, "Bangalore": 980, "Delhi": 1400 }
    };

    return cityDistances[departure] ? cityDistances[departure][destination] : null;
}





document.getElementById("flightForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let formData = new FormData(this);

    
    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById("predict").textContent = data.predicted_price;
            document.getElementById("predictionResult").classList.remove("hidden");
        }
    })
    .catch(error => console.error("Error:", error));
});



